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
"""Backend-neutral description of a trainer rank's piece of an HF tensor.

Point-to-point checkpoint engines (``nccl_m2n``, ``sharded_nixl``) consume shard sources that yield
``(hf_name, local_view, HFShardMeta)``: the view is this rank's dense box of the full HF tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["HFShardMeta", "dense_box"]


@dataclass(frozen=True)
class HFShardMeta:
    """Where a local HF shard sits in its full tensor."""

    name: str
    global_shape: tuple[int, ...]
    dtype: torch.dtype
    # start of this rank's box in the full tensor, and the box size (= local view shape)
    offset: tuple[int, ...]
    local_shape: tuple[int, ...]

    @property
    def is_full(self) -> bool:
        return self.local_shape == self.global_shape

    def shard_dim(self) -> Optional[int]:
        """The single dim this box is cut on (``None`` for the full tensor); raises if cut on several."""
        dims = [d for d, (n, g) in enumerate(zip(self.local_shape, self.global_shape, strict=True)) if n != g]
        if not dims:
            return None
        if len(dims) > 1:
            raise ValueError(f"{self.name}: box {self.offset}+{self.local_shape} is cut on dims {dims}")
        return dims[0]


def dense_box(flat_pos: torch.Tensor, shape: tuple[int, ...]) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """``(offset, size)`` of the box ``flat_pos`` covers in ``shape``, or ``None`` if the positions
    are not exactly one dense box in ascending row-major order."""
    n = flat_pos.numel()
    if n == 0:
        return None
    coords = torch.stack(torch.unravel_index(flat_pos.long(), tuple(shape)), dim=1)
    lo = coords.min(0).values
    size = coords.max(0).values - lo + 1
    if int(size.prod()) != n:
        return None
    # dense and ascending: every position equals the row-major enumeration of the box
    strides = torch.ones(len(shape), dtype=torch.int64, device=flat_pos.device)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    start = int((lo * strides).sum())
    local = torch.arange(n, device=flat_pos.device)
    expect = torch.zeros(n, dtype=torch.int64, device=flat_pos.device)
    rem = local
    for d in range(len(shape) - 1, -1, -1):
        expect += (rem % size[d]) * strides[d]
        rem = rem // size[d]
    if not torch.equal(flat_pos.long() - start, expect):
        return None
    return tuple(lo.tolist()), tuple(size.tolist())
