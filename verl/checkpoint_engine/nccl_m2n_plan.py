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
"""Transfer plan of the ``nccl_m2n`` checkpoint engine (no NCCL here, CPU-testable).

One channel moves the tensors of one trainer pipeline stage to one rollout replica over its own
M2N communicator: comm ranks ``[0, S)`` are the stage's senders in sender order, ``[S, S + D)``
the replica's rollout ranks in TP order. Every tensor becomes a :class:`PlanItem` whose two
:class:`Side` s are what ``nccl.m2n.DistTensor`` needs on every rank of the channel:

* source: from the senders' manifests (``(name, global_shape, dtype, offset, local_shape)`` per
  piece). The holders must be a contiguous run of senders; their boxes are either all the full
  tensor (``Replicate``) or an even split on one dim in sender order (``Shard(d)``). M2N has one
  local shape per side, so uneven splits must be resolved by the source (gather that tensor).
* destination: from the replica's ``RolloutLayout.slices`` boxes. A tensor is ``Shard(d)`` only if
  every rank's box is the even split on ``d`` in TP order, otherwise ``Replicate`` (every rank
  receives the full tensor and its loader slices it).

The plan only depends on the manifests and layouts, so trainer and rollout ranks build it on their
own and compare a digest instead of shipping per-weight metadata on every step.

MoE experts may travel stacked per layer under the transformers-v5 names (one call per layer instead
of one per expert): ``<layer>.mlp.experts.gate_up_proj`` described as ``[2E, I, H]`` (row ``2e`` is
expert ``e``'s ``gate_proj``, ``2e + 1`` its ``up_proj``) and ``<layer>.mlp.experts.down_proj`` as
``[E, H, I]``. Their destination boxes are derived from the per-expert ``slices``, and receivers
turn them back into per-expert views with :func:`unstack`.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

__all__ = [
    "Piece",
    "Side",
    "PlanItem",
    "build_channel_plan",
    "plan_digest",
    "split_buckets",
    "stacked_kind",
    "expert_names",
    "unstack",
    "MAX_TENSOR_DIMS",
]

_STACKED = re.compile(r"^(?P<prefix>.+)\.mlp\.experts\.(?P<kind>gate_up_proj|down_proj)$")

MAX_TENSOR_DIMS = 3
# M2N v0.2: at most 16 source ranks feed one destination shard
MAX_SOURCES_PER_DST = 16

_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}


@dataclass(frozen=True)
class Piece:
    """One sender's piece of an HF tensor (its dense box in the full tensor)."""

    name: str
    global_shape: tuple[int, ...]
    dtype: str
    offset: tuple[int, ...]
    local_shape: tuple[int, ...]


@dataclass(frozen=True)
class Side:
    """One side of a reshard: ranks ``[start, start + n)`` of the channel comm."""

    start: int
    n: int
    # None: every rank of the side holds the full tensor
    shard_dim: Optional[int]
    local_shape: tuple[int, ...]


@dataclass(frozen=True)
class PlanItem:
    name: str
    global_shape: tuple[int, ...]
    dtype: str
    src: Side
    dst: Side

    @property
    def dst_bytes(self) -> int:
        return math.prod(self.dst.local_shape) * _DTYPE_BYTES[self.dtype]


def _even_split(boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]], shape: tuple[int, ...]) -> Optional[int]:
    """The dim ``d`` on which ``boxes`` (in rank order) are ``torch.chunk``-even pieces of ``shape``,
    i.e. box ``k`` is ``[k * shape[d] / n, (k + 1) * shape[d] / n)`` and full on every other dim.
    ``-1`` if every box is the full tensor; ``None`` otherwise."""
    n = len(boxes)
    if all(tuple(size) == tuple(shape) and not any(off) for off, size in boxes):
        return -1
    cut = {d for off, size in boxes for d, (o, s) in enumerate(zip(off, size, strict=True)) if o or s != shape[d]}
    if len(cut) != 1:
        return None
    d = cut.pop()
    if shape[d] % n:
        return None
    step = shape[d] // n
    for k, (off, size) in enumerate(boxes):
        if off[d] != k * step or size[d] != step:
            return None
    return d


def _local(shape: tuple[int, ...], d: Optional[int], n: int) -> tuple[int, ...]:
    if d is None:
        return tuple(shape)
    out = list(shape)
    out[d] //= n
    return tuple(out)


def build_channel_plan(
    manifests: Sequence[Sequence[Piece]],
    dst_slices: Sequence[Optional[dict]],
) -> list[PlanItem]:
    """Plan one channel.

    Args:
        manifests: per sender of the stage (sender order), the pieces it ships, in its yield order.
        dst_slices: per rollout rank of the replica (TP order), its ``RolloutLayout.slices``
            (``{hf_name: {"full_shape": [...], "box": [[start, len], ...]}}``) or ``None`` (full).

    Returns:
        Items in a deterministic order: first appearance over the senders' manifests.
    """
    holders: dict[str, list[tuple[int, Piece]]] = {}
    order: list[str] = []
    for s, pieces in enumerate(manifests):
        for p in pieces:
            if p.name not in holders:
                holders[p.name] = []
                order.append(p.name)
            holders[p.name].append((s, p))

    S, D = len(manifests), len(dst_slices)
    items = []
    for name in order:
        hs = holders[name]
        first = hs[0][1]
        shape, dtype = tuple(first.global_shape), first.dtype
        if len(shape) == 0 or len(shape) > MAX_TENSOR_DIMS:
            raise ValueError(f"{name}: M2N moves 1-D to {MAX_TENSOR_DIMS}-D tensors, got shape {shape}")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"{name}: dtype {dtype} is not supported by M2N")
        ranks = [s for s, _ in hs]
        if ranks != list(range(ranks[0], ranks[0] + len(ranks))):
            raise ValueError(f"{name}: held by senders {ranks}, which are not a contiguous run")
        for _s, p in hs:
            if tuple(p.global_shape) != shape or p.dtype != dtype:
                raise ValueError(f"{name}: senders disagree on shape/dtype ({p.global_shape}, {p.dtype})")
        sd = _even_split([(tuple(p.offset), tuple(p.local_shape)) for _, p in hs], shape)
        if sd is None:
            raise ValueError(
                f"{name}: sender boxes {[(p.offset, p.local_shape) for _, p in hs]} are neither full nor an even "
                "split on one dim; the shard source must gather this tensor"
            )
        src_dim = None if sd < 0 else sd
        src = Side(ranks[0], len(ranks), src_dim, _local(shape, src_dim, len(ranks)))

        dst_dim = _dst_dim(name, shape, dst_slices)
        dst = Side(S, D, dst_dim, _local(shape, dst_dim, D))
        if src_dim is not None and (dst_dim is None or dst_dim != src_dim) and src.n > MAX_SOURCES_PER_DST:
            raise ValueError(
                f"{name}: {src.n} source shards feed every destination shard, M2N allows {MAX_SOURCES_PER_DST}"
            )
        items.append(PlanItem(name, shape, dtype, src, dst))
    return items


def _dst_dim(name: str, shape: tuple[int, ...], dst_slices: Sequence[Optional[dict]]) -> Optional[int]:
    boxes = []
    for slices in dst_slices:
        entry = (slices or {}).get(name) or _stacked_entry(name, shape, slices or {})
        if entry is None or tuple(entry["full_shape"]) != shape:
            return None
        boxes.append((tuple(int(b[0]) for b in entry["box"]), tuple(int(b[1]) for b in entry["box"])))
    sd = _even_split(boxes, shape) if boxes else None
    return None if sd is None or sd < 0 else sd


def stacked_kind(name: str) -> Optional[tuple[str, str]]:
    """``(layer prefix, "gate_up_proj" | "down_proj")`` for a stacked expert name, else ``None``."""
    m = _STACKED.match(name)
    return (m["prefix"], m["kind"]) if m else None


def expert_names(prefix: str, kind: str, e: int) -> list[str]:
    base = f"{prefix}.mlp.experts.{e}."
    if kind == "gate_up_proj":
        return [base + "gate_proj.weight", base + "up_proj.weight"]
    return [base + "down_proj.weight"]


def _stacked_entry(name: str, shape: tuple[int, ...], slices: dict) -> Optional[dict]:
    """The box of a stacked expert tensor on one rollout rank, if every expert's per-expert boxes are
    the same (each expert sliced alike, as TP does); ``None`` (full) otherwise."""
    sk = stacked_kind(name)
    if sk is None or len(shape) != 3:
        return None
    prefix, kind = sk
    n_rows = 2 if kind == "gate_up_proj" else 1
    box = None
    for e in range(shape[0] // n_rows):
        for n in expert_names(prefix, kind, e):
            entry = slices.get(n)
            if entry is None or tuple(entry["full_shape"]) != tuple(shape[1:]):
                return None
            b = tuple((int(s), int(c)) for s, c in entry["box"])
            if box is None:
                box = b
            elif b != box:
                return None
    return {"full_shape": list(shape), "box": [[0, shape[0]], *[list(b) for b in box]]}


def unstack(item: PlanItem, tensor, rank: int) -> list[tuple[str, object]]:
    """Per-expert views of a received stacked expert tensor (``rank``: this receiver's index on the
    destination side). Other items come back as ``[(name, tensor)]``."""
    sk = stacked_kind(item.name)
    if sk is None or len(item.global_shape) != 3:
        return [(item.name, tensor)]
    prefix, kind = sk
    n_rows = 2 if kind == "gate_up_proj" else 1
    first = rank * item.dst.local_shape[0] if item.dst.shard_dim == 0 else 0
    out = []
    for row in range(tensor.shape[0]):
        e, part = divmod(first + row, n_rows)
        out.append((expert_names(prefix, kind, e)[part], tensor[row]))
    return out


def plan_digest(items: Iterable[PlanItem]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(repr((it.name, it.global_shape, it.dtype, it.src, it.dst)).encode())
    return h.hexdigest()[:16]


def split_buckets(items: Sequence[PlanItem], bucket_bytes: int, group_bytes: int = 0) -> list[list[list[PlanItem]]]:
    """Cut ``items`` into receive buckets of at most ``bucket_bytes`` destination bytes (a larger
    item gets a bucket of its own), each a list of ``m2n.group`` batches of at most ``group_bytes``
    (0: one item per batch). Both sides call this with the same arguments so the grouped calls match.
    """
    buckets: list[list[PlanItem]] = []
    cur, cur_b = [], 0
    for it in items:
        b = it.dst_bytes
        if cur and cur_b + b > bucket_bytes:
            buckets.append(cur)
            cur, cur_b = [], 0
        cur.append(it)
        cur_b += b
    if cur:
        buckets.append(cur)
    out = []
    for bucket in buckets:
        if group_bytes <= 0:
            out.append([[it] for it in bucket])
            continue
        groups, g, g_b = [], [], 0
        for it in bucket:
            if g and g_b + it.dst_bytes > group_bytes:
                groups.append(g)
                g, g_b = [], 0
            g.append(it)
            g_b += it.dst_bytes
        groups.append(g)
        out.append(groups)
    return out
