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
"""HF-coordinate local shards of an FSDP2 model, for point-to-point checkpoint engines.

FSDP2 already keeps every parameter under its HF name as a DTensor, so a rank's piece is just
``to_local()`` and its box comes from the DTensor placements. A shard dim that does not split
evenly (``torch.chunk`` semantics: the last ranks are shorter or empty) cannot be described by one
M2N local shape; that tensor alone is gathered with ``full_tensor()`` and shipped full.

Under HSDP only the ranks at coordinate 0 of every replicate mesh dim send; the others still take
part in the gathers.

transformers-v5 MoE models already keep their experts stacked per layer; they go out in the
``nccl_m2n`` stacked-expert format (:mod:`verl.checkpoint_engine.nccl_m2n_plan`).
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from verl.utils.device import get_device_id
from verl.utils.model import convert_weight_keys

from ..hf_shard import HFShardMeta

__all__ = ["FSDPHFShardSource"]


def _even(param: DTensor) -> bool:
    mesh = param.device_mesh
    for i, pl in enumerate(param.placements):
        if type(pl) is Shard:
            if param.shape[pl.dim] % mesh.size(i):
                return False
        elif not isinstance(pl, Replicate):
            # _StridedShard, Partial: not a single dense box
            return False
    return True


class FSDPHFShardSource:
    """Local HF shards of an FSDP2 :class:`~verl.workers.engine.fsdp.transformer_impl.FSDPEngine`."""

    stage = 0

    def __init__(self, engine):
        from verl.utils.fsdp_utils import fsdp_version

        self.engine = engine
        module = engine.module
        if fsdp_version(module) != 2:
            raise NotImplementedError("HF shard export needs FSDP2 (actor.strategy=fsdp2)")
        if hasattr(getattr(module, "_fsdp_wrapped_module", module), "peft_config"):
            raise NotImplementedError("HF shard export does not support LoRA yet")
        self.sender_index: Optional[int] = None
        dt = next((p for p in module.state_dict().values() if isinstance(p, DTensor)), None)
        if dt is None:
            self.sender_index = torch.distributed.get_rank()
            return
        mesh, coord = dt.device_mesh, dt.device_mesh.get_coordinate()
        shard_dims = [i for i, pl in enumerate(dt.placements) if isinstance(pl, Shard)]
        if any(coord[i] for i, pl in enumerate(dt.placements) if not isinstance(pl, Shard)):
            return
        idx = 0
        for i in shard_dims:
            idx = idx * mesh.size(i) + coord[i]
        self.sender_index = idx

    def iter_shards(self) -> Iterator[tuple[str, torch.Tensor, HFShardMeta]]:
        """Yield ``(hf_name, local_view, meta)`` in state-dict order (identical on every rank).
        Gathers of unevenly sharded tensors are collective over the FSDP mesh."""
        module = self.engine.module
        params = convert_weight_keys(module.state_dict(), getattr(module, "_fsdp_wrapped_module", module))
        device = get_device_id()
        for name, param in params.items():
            if isinstance(param, DTensor) and _even(param):
                local = param.to_local().to(device, non_blocking=True)
                shape, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
                meta = HFShardMeta(name, tuple(param.shape), local.dtype, tuple(offset), tuple(shape))
            else:
                if isinstance(param, DTensor):
                    local = param.to(device, non_blocking=True).full_tensor()
                else:
                    local = param.to(device, non_blocking=True)
                meta = HFShardMeta(name, tuple(local.shape), local.dtype, (0,) * local.dim(), tuple(local.shape))
            if self.sender_index is None:
                continue
            if local.is_floating_point() and local.dtype != torch.bfloat16:
                local = local.to(torch.bfloat16)
            meta = _restack(HFShardMeta(meta.name, meta.global_shape, local.dtype, meta.offset, meta.local_shape))
            yield name, local.reshape(meta.local_shape), meta


def _restack(meta: HFShardMeta) -> HFShardMeta:
    """transformers-v5 stacked ``gate_up_proj`` ``[E, 2I, H]`` (gate rows, then up rows, per expert)
    as the ``[2E, I, H]`` of the ``nccl_m2n`` stacked-expert format. A box cut on the expert dim only
    stays a box."""
    from verl.checkpoint_engine.nccl_m2n_plan import stacked_kind

    sk = stacked_kind(meta.name)
    if sk is None or sk[1] != "gate_up_proj" or len(meta.global_shape) != 3:
        return meta
    (n_e, two_i, h), (o_e, o_i, o_h), (l_e, l_i, l_h) = meta.global_shape, meta.offset, meta.local_shape
    if two_i % 2 or o_i or o_h or l_i != two_i or l_h != h:
        raise ValueError(f"{meta.name}: box {meta.offset}+{meta.local_shape} splits the gate/up rows of an expert")
    return HFShardMeta(meta.name, (2 * n_e, two_i // 2, h), meta.dtype, (2 * o_e, 0, 0), (2 * l_e, two_i // 2, h))
