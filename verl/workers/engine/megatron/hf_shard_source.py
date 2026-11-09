# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HF-coordinate local shards straight from Megatron, without communication.

Point-to-point checkpoint engines (``nccl_m2n``, ``sharded_nixl``) need, for every
HF tensor, the part this trainer rank holds and where it sits in the full tensor.
:class:`MegatronHFShardSource` derives both from the comm-stubbed probe in
:mod:`.delta_export`:

* one-time :meth:`plan`: push an int64 position marker through each mcore
  parameter's probe (cached as source runs) and turn every HF slot's positions
  into a dense box ``(offset, local_shape)`` in the full HF tensor;
* steady state :meth:`iter_shards`: slice the cached runs and yield
  ``(hf_name, local_view, HFShardMeta)``; no probe, no collective.

A slot whose positions are not a dense box on some rank (uneven vocab tails,
strided layouts) falls back for that tensor only: the ranks of its merge group
all-gather their pieces and every one of them yields the full tensor as a
``Replicate`` box. The decision is agreed over WORLD at plan time so all ranks
stay in lockstep.

Senders: the ranks of a pipeline stage that ship anything (dp/cp and expert-dp
replicas are deduplicated by the export index), numbered in global rank order.
Megatron keeps TP (and ETP) ranks adjacent, so the holders of every HF tensor
are a contiguous run of senders, as M2N source meshes need.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.distributed as dist

from ..hf_shard import HFShardMeta, dense_box

logger = logging.getLogger(__name__)

__all__ = ["HFShardMeta", "MegatronHFShardSource", "dense_box"]


@dataclass
class _Row:
    rec: object  # McoreParamExport
    # per HF slot: (meta, count) for box slots; None when this row falls back
    metas: Optional[list] = None
    fallback: bool = False


class MegatronHFShardSource:
    """Local HF shards of a :class:`~verl.workers.engine.megatron.transformer_impl.MegatronEngine`.

    Args:
        engine: the Megatron engine (``_mcore_export_index`` / ``_delta_slot_cache``).
        force_fallback: HF names to export through the gather fallback (tests).
    """

    def __init__(self, engine, force_fallback: Optional[set[str]] = None):
        self.engine = engine
        self.force_fallback = set(force_fallback or ())
        self._rows: Optional[list[_Row]] = None
        from megatron.core import parallel_state as mpu

        self._dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        self.stage = mpu.get_pipeline_model_parallel_rank()
        # index among the senders of this stage, None if this rank ships nothing; set by plan()
        self.sender_index: Optional[int] = None

    @property
    def planned(self) -> bool:
        return self._rows is not None

    def _entry(self, rec, lidx, lval):
        from .delta_export import mcore_hf_delta_entry

        return mcore_hf_delta_entry(rec, 0, lidx, lval, self.engine._delta_slot_cache)

    def _local(self, rec) -> Optional[torch.Tensor]:
        """This rank's flat bf16 shard of ``rec``, or None if it does not ship it."""
        if rec.param is None or not rec.spec.contributes:
            return None
        if rec.spec.gather_group is None and self._dp_rank != 0:
            # replicated params: only the dp/cp rank 0 copy ships (all tp ranks hold it)
            return None
        local = rec.param.data
        if local.is_floating_point() and local.dtype != torch.bfloat16:
            local = local.to(torch.bfloat16)
        return local.reshape(-1)

    def plan(self) -> None:
        """One-time box discovery (1-3 s on real models). Collective over WORLD."""
        index = self.engine._mcore_export_index()
        rows, ok = [], []
        for rec in index:
            row = _Row(rec)
            lval = self._local(rec)
            good = True
            if lval is not None:
                lidx = torch.arange(lval.numel(), dtype=torch.int64, device=lval.device)
                slots, _dt, counts, hf_idx, _hf_val = self._entry(rec, lidx, lval)
                row.metas, off = [], 0
                for (name, shape), c in zip(slots, counts.tolist(), strict=True):
                    box = dense_box(hf_idx[off : off + c], tuple(shape)) if c else None
                    off += c
                    if c and (box is None or name in self.force_fallback):
                        good = False
                    meta = None if not c else HFShardMeta(name, tuple(shape), lval.dtype, *box) if box else None
                    row.metas.append((meta, c))
                # warm the cached source runs used by iter_shards
                self._entry(rec, None, lval)
                if rec.shard_runs is None or rec.shard_runs[1] is None:
                    good = False
            rows.append(row)
            ok.append(good)
        # a row falls back on every rank if it is not a box on any rank
        flags = torch.tensor(ok, dtype=torch.int32, device=torch.cuda.current_device())
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        for row, flag in zip(rows, flags.tolist(), strict=True):
            row.fallback = not flag
            if row.fallback:
                logger.warning("%s: HF output is not a dense box, exported via gather", row.rec.megatron_name)
        self._rows = rows
        ships = any(
            self._local(row.rec) is not None and (row.fallback or any(c for _m, c in row.metas)) for row in rows
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            every: list = [None] * dist.get_world_size()
            dist.all_gather_object(every, (self.stage, ships))
            senders = [r for r, (st, s) in enumerate(every) if st == self.stage and s]
            self.sender_index = senders.index(dist.get_rank()) if ships else None
        else:
            self.sender_index = 0 if ships else None

    def fallback_names(self) -> list[str]:
        return [r.rec.megatron_name for r in self._rows or () if r.fallback]

    def iter_shards(self) -> Iterator[tuple[str, torch.Tensor, HFShardMeta]]:
        """Yield ``(hf_name, local_view, meta)`` for every HF tensor piece this rank ships, in an
        order identical on every rank of a pipeline stage. Gather fallbacks are collective."""
        if self._rows is None:
            self.plan()
        for row in self._rows:
            rec, lval = row.rec, self._local(row.rec)
            if row.fallback:
                yield from self._gather_row(rec, lval)
                continue
            if lval is None:
                continue
            _slots, _dt, _counts, _idx, hf_val = self._entry(rec, None, lval)
            off = 0
            for meta, c in row.metas:
                if c:
                    yield meta.name, hf_val[off : off + c].view(meta.local_shape), meta
                off += c

    def _gather_row(self, rec, lval):
        """Fallback: all-gather this row's (positions, values) over its merge group and yield
        full tensors. Every rank of the group takes part, also non-contributing ones."""
        group = rec.spec.gather_group
        dev = torch.cuda.current_device()
        if lval is not None:
            lidx = torch.arange(lval.numel(), dtype=torch.int64, device=lval.device)
            slots, _dt, counts, hf_idx, hf_val = self._entry(rec, lidx, lval)
        else:
            slots = rec.slots
            counts = torch.zeros(len(slots or ()), dtype=torch.int64)
            hf_idx = torch.empty(0, dtype=torch.int64, device=dev)
            hf_val = torch.empty(0, dtype=torch.bfloat16, device=dev)
        if group is None or dist.get_world_size(group) == 1:
            pieces = [(counts.cpu(), hf_idx, hf_val)]
        else:
            n = dist.get_world_size(group)
            cnt = counts.to(dev, torch.int64)
            all_cnt = [torch.empty_like(cnt) for _ in range(n)]
            dist.all_gather(all_cnt, cnt, group=group)
            width = max(int(c.sum()) for c in all_cnt)
            pad_idx = torch.zeros(width, dtype=torch.int64, device=dev)
            pad_val = torch.zeros(width, dtype=hf_val.dtype, device=dev)
            pad_idx[: hf_idx.numel()] = hf_idx
            pad_val[: hf_val.numel()] = hf_val
            all_idx = [torch.empty_like(pad_idx) for _ in range(n)]
            all_val = [torch.empty_like(pad_val) for _ in range(n)]
            dist.all_gather(all_idx, pad_idx, group=group)
            dist.all_gather(all_val, pad_val, group=group)
            pieces = [(c.cpu(), i, v) for c, i, v in zip(all_cnt, all_idx, all_val, strict=True)]
        if lval is None:
            return
        for k, (name, shape) in enumerate(slots):
            full = torch.empty(tuple(shape), dtype=hf_val.dtype, device=dev)
            flat = full.view(-1)
            for c, i, v in pieces:
                start = int(c[:k].sum())
                end = start + int(c[k])
                flat[i[start:end]] = v[start:end]
            yield name, full, HFShardMeta(name, tuple(shape), full.dtype, (0,) * len(shape), tuple(shape))
