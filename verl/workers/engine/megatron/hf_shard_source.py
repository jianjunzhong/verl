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

With a rollout weight quantizer the pieces are quantized locally
(:mod:`..hf_shard_quant`); a tensor whose box cuts a quantization block on
some rank takes the gather fallback and is quantized whole.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
import torch.distributed as dist

from ..hf_shard import HFShardMeta, dense_box

logger = logging.getLogger(__name__)

__all__ = ["HFShardMeta", "MegatronHFShardSource", "dense_box"]

_EXPERT = re.compile(r"^(?P<prefix>.+)\.mlp\.experts\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")


@dataclass
class _Row:
    rec: object  # McoreParamExport
    # per HF slot: (meta, count) for box slots; None when this row falls back
    metas: Optional[list] = None
    fallback: bool = False
    # this rank's piece cuts quantization blocks (rollout quantization only)
    cuts_blocks: bool = False


def _stacked_meta(key, n_e, e0, n, off, size, shape) -> HFShardMeta:
    """The stacked box of local experts ``[e0, e0 + n)`` whose per-expert box is ``(off, size)`` in
    a per-expert tensor of ``shape``: ``gate_up_proj`` ``[2E, I, H]``, ``down_proj`` ``[E, H, I]``."""
    prefix, kind = key
    rows = 2 if kind == "gate_up_proj" else 1
    return HFShardMeta(
        f"{prefix}.mlp.experts.{kind}",
        (rows * n_e, *shape),
        torch.bfloat16,
        (rows * e0, *off),
        (rows * n, *size),
    )


def _cut(kind, n_e, e0, n, box) -> list[int]:
    (off, size), shape = box
    meta = _stacked_meta(("", kind), n_e, e0, n, off, size, shape)
    return [d for d, (s, g) in enumerate(zip(meta.local_shape, meta.global_shape, strict=True)) if s != g]


@dataclass
class _Stack:
    """One layer's experts of one projection kind, stacked in a persistent buffer."""

    meta: HFShardMeta
    # (row index, offset of the slot in the row's HF values, count, buffer row)
    parts: list = field(default_factory=list)
    buf: Optional[torch.Tensor] = None


class MegatronHFShardSource:
    """Local HF shards of a :class:`~verl.workers.engine.megatron.transformer_impl.MegatronEngine`.

    Args:
        engine: the Megatron engine (``_mcore_export_index`` / ``_delta_slot_cache``).
        force_fallback: HF names to export through the gather fallback (tests).
        stack_experts: ship each layer's local MoE experts as one stacked tensor per projection
            kind (:mod:`verl.checkpoint_engine.nccl_m2n_plan`), copied into a persistent buffer every
            step, wherever the stacked box is cut on at most one dim on every rank (so not with
            EP > 1 and ETP > 1 together). One expert-dp replica ships each stack.
        quantizer: the engine's rollout weight quantizer, applied to the local pieces.
    """

    def __init__(self, engine, force_fallback: Optional[set[str]] = None, stack_experts: bool = False, quantizer=None):
        from ..hf_shard_quant import LocalShardQuantizer

        self.engine = engine
        self.force_fallback = set(force_fallback or ())
        self.stack_experts = stack_experts
        self.quant = LocalShardQuantizer(quantizer) if quantizer is not None else None
        self._rows: Optional[list[_Row]] = None
        self._stacks: list[_Stack] = []
        self._stacked_rows: set[int] = set()
        from megatron.core import parallel_state as mpu

        self._dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        self._edp = (mpu.get_expert_data_parallel_rank(), mpu.get_expert_data_parallel_world_size())
        self.stage = mpu.get_pipeline_model_parallel_rank()
        # index among the senders of this stage, None if this rank ships nothing; set by plan()
        self.sender_index: Optional[int] = None

    @property
    def planned(self) -> bool:
        return self._rows is not None

    def _entry(self, rec, lidx, lval):
        from .delta_export import mcore_hf_delta_entry

        return mcore_hf_delta_entry(rec, 0, lidx, lval, self.engine._delta_slot_cache)

    def _local(self, rec, any_replica: bool = False) -> Optional[torch.Tensor]:
        """This rank's flat bf16 shard of ``rec``, or None if it does not ship it (``any_replica``:
        also when another replica ships it)."""
        if rec.param is None:
            return None
        if not any_replica and not rec.spec.contributes:
            return None
        if not any_replica and rec.spec.gather_group is None and self._dp_rank != 0:
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
                    elif c and self.quant is not None and not self.quant.aligned(name, tuple(shape), *box):
                        good, row.cuts_blocks = False, True
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
        for row, flag, good in zip(rows, flags.tolist(), ok, strict=True):
            row.fallback = not flag
            if not good:  # logged by the ranks that cause the fallback
                why = "its box cuts quantization blocks" if row.cuts_blocks else "HF output is not a dense box"
                logger.warning("%s: %s, exported via gather", row.rec.megatron_name, why)
        self._rows = rows
        if self.stack_experts:
            self._plan_stacks(rows)
        ships = bool(self._stacks) or any(
            i not in self._stacked_rows
            and self._local(row.rec) is not None
            and (row.fallback or any(c for _m, c in row.metas))
            for i, row in enumerate(rows)
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            every: list = [None] * dist.get_world_size()
            dist.all_gather_object(every, (self.stage, ships))
            senders = [r for r, (st, s) in enumerate(every) if st == self.stage and s]
            self.sender_index = senders.index(dist.get_rank()) if ships else None
        else:
            self.sender_index = 0 if ships else None

    def _row_boxes(self, row: _Row) -> list[tuple]:
        """``(hf_name, full_shape, value offset, count, box)`` of every slot of ``row`` this rank holds
        (on any replica)."""
        full = self._local(row.rec, any_replica=True)
        if full is None or row.fallback:
            return []
        lidx = torch.arange(full.numel(), dtype=torch.int64, device=full.device)
        slots, _dt, counts, hf_idx, _hf_val = self._entry(row.rec, lidx, full)
        out, off = [], 0
        for (name, shape), c in zip(slots, counts.tolist(), strict=True):
            if c:
                out.append((name, tuple(shape), off, c, dense_box(hf_idx[off : off + c], tuple(shape))))
            off += c
        return out

    def _plan_stacks(self, rows: list[_Row]) -> None:
        """Choose the expert groups that travel stacked (the same choice on every rank) and the
        buffers this rank fills. Collective over WORLD."""
        found: dict = {}  # (prefix, kind) -> {expert: {proj: (row, value offset, count, box, shape)}}
        for i, row in enumerate(rows):
            slots = self._row_boxes(row)
            hits = [(_EXPERT.match(n), s, off, c, box) for n, s, off, c, box in slots]
            if not hits or not all(m and box for m, _s, _o, _c, box in hits):
                continue  # only rows that are entirely expert boxes
            for m, shape, off, c, box in hits:
                kind = "down_proj" if m["proj"] == "down_proj" else "gate_up_proj"
                found.setdefault((m["prefix"], kind), {}).setdefault(int(m["e"]), {})[m["proj"]] = (
                    i,
                    off,
                    c,
                    box,
                    shape,
                )
        local = {}
        for key, experts in found.items():
            projs = ("gate_proj", "up_proj") if key[1] == "gate_up_proj" else ("down_proj",)
            es = sorted(experts)
            boxes = {(experts[e][p][3], experts[e][p][4]) for e in es for p in projs if p in experts[e]}
            ok = es == list(range(es[0], es[-1] + 1)) and all(set(experts[e]) == set(projs) for e in es)
            ok = ok and len(boxes) == 1
            local[key] = (ok, es[0], len(es), next(iter(boxes)) if ok else None)
        every = [local]
        if dist.is_initialized() and dist.get_world_size() > 1:
            every = [None] * dist.get_world_size()
            dist.all_gather_object(every, local)

        keys = sorted({k for d in every for k in d})
        chosen = []
        for key in keys:
            entries = [d[key] for d in every if key in d]
            n_e = max(e0 + n for _ok, e0, n, _b in entries)
            covered = {e for _ok, e0, n, _b in entries for e in range(e0, e0 + n)}
            if not all(ok for ok, *_ in entries) or covered != set(range(n_e)):
                continue
            if all(len(_cut(key[1], n_e, e0, n, b)) <= 1 for _ok, e0, n, b in entries):
                chosen.append((key, n_e))
        edp_rank, edp_world = self._edp
        for k, (key, n_e) in enumerate(chosen):
            experts = found.get(key)
            if experts is None:
                continue
            self._stacked_rows.update(part[0] for parts in experts.values() for part in parts.values())
            if k % edp_world != edp_rank:
                continue  # another expert-dp replica ships this stack
            _ok, e0, n, ((off, size), shape) = local[key]
            meta = _stacked_meta(key, n_e, e0, n, off, size, shape)
            stack = _Stack(meta, buf=torch.empty(meta.local_shape, dtype=torch.bfloat16, device="cuda"))
            projs = ("gate_proj", "up_proj") if key[1] == "gate_up_proj" else ("down_proj",)
            for e in range(e0, e0 + n):
                for j, p in enumerate(projs):
                    i, voff, c, _box, _shape = experts[e][p]
                    stack.parts.append((i, voff, c, len(projs) * (e - e0) + j))
            self._stacks.append(stack)
        if chosen:
            logger.info("stacked %d of %d expert groups", len(chosen), len(keys))

    def fallback_names(self) -> list[str]:
        return [r.rec.megatron_name for r in self._rows or () if r.fallback]

    def iter_shards(self) -> Iterator[tuple[str, torch.Tensor, HFShardMeta]]:
        """Yield ``(hf_name, local_view, meta)`` for every HF tensor piece this rank ships, in an
        order identical on every rank of a pipeline stage. Gather fallbacks are collective."""
        if self._rows is None:
            self.plan()
        if self.quant is None:
            yield from self._iter_bf16()
        else:
            yield from self.quant(list(self._iter_bf16()))

    def _iter_bf16(self) -> Iterator[tuple[str, torch.Tensor, HFShardMeta]]:
        fills: dict[int, list] = {}
        for stack in self._stacks:
            for i, voff, c, brow in stack.parts:
                fills.setdefault(i, []).append((stack.buf[brow], voff, c))
        for i, row in enumerate(self._rows):
            rec, lval = row.rec, self._local(row.rec)
            if i in self._stacked_rows:
                if i in fills:
                    hf_val = self._entry(rec, None, self._local(rec, any_replica=True))[4]
                    for dst, voff, c in fills[i]:
                        dst.copy_(hf_val[voff : voff + c].view(dst.shape))
                continue
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
        for stack in self._stacks:
            yield stack.meta.name, stack.buf, stack.meta

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
