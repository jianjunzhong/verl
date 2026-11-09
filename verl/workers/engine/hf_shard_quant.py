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
"""Trainer-side rollout quantization of local HF shards, for point-to-point checkpoint engines.

verl's rollout weight quantizers (:func:`verl.utils.qat.dispatch_rollout_weight_quantizer`) turn
full HF tensors into the rollout's ``(weight, scale, ...)`` tensors. :class:`LocalShardQuantizer`
runs the same quantizer on each rank's local shard and derives the box of every output from the
weight's box, so the outputs are bitwise the slices of quantizing the full tensor, provided that

* every cut of the shard falls on a quantization block boundary (or the end of the tensor), and
* outputs not shaped like the weight (per-tensor global scales) come from an uncut weight.

:meth:`LocalShardQuantizer.aligned` tells the shard source which pieces break that, so it gathers
those tensors and quantizes them whole. Block factors are learned once per name by quantizing a
probe tensor. Stacked experts (:mod:`verl.checkpoint_engine.nccl_m2n_plan`) go through the
quantizer's packed-expert path and are restacked per output; schemes without one (per-token NVFP4)
are quantized expert by expert.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import torch

from verl.checkpoint_engine.nccl_m2n_plan import expert_names, stacked_kind

from .hf_shard import HFShardMeta

__all__ = ["LocalShardQuantizer"]

_PROBE = 512
_EXPERT_OUT = re.compile(r"^.+\.mlp\.experts\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<key>\w+)$")
_GATE_UP = re.compile(r"^(?P<base>.+\.experts\.\d+)\.(?:gate_proj|up_proj)\.weight$")


class _Cut(ValueError):
    pass


def _module(name: str) -> str:
    return name.rsplit(".", 1)[0]


def _group(name: str) -> list[str]:
    """The names a quantizer may need together with ``name`` (per-token NVFP4 pairs gate and up)."""
    m = _GATE_UP.match(name)
    return [f"{m['base']}.gate_proj.weight", f"{m['base']}.up_proj.weight"] if m else [name]


def _box(factors: Optional[tuple[int, ...]], meta: HFShardMeta):
    """``(global_shape, offset, local_shape)`` of an output with per-dim block ``factors`` (``None``:
    not shaped like the weight) of the weight piece ``meta``; :class:`_Cut` if it cannot be local."""
    g, off, size = meta.global_shape, meta.offset, meta.local_shape
    if factors is None:
        if tuple(size) != tuple(g):
            raise _Cut(f"{meta.name}: a per-tensor output needs the whole weight")
        return None
    out_g, out_off, out_size = [], [], []
    for d, (n, o, s, k) in enumerate(zip(g, off, size, factors, strict=True)):
        if s != n and (o % k or (s % k and o + s != n)):
            raise _Cut(f"{meta.name}: the cut [{o}, {o + s}) of dim {d} splits {k}-wide quantization blocks")
        out_g.append(-(-n // k))
        out_off.append(o // k)
        out_size.append(-(-s // k))
    return tuple(out_g), tuple(out_off), tuple(out_size)


class LocalShardQuantizer:
    """Wraps a rollout weight quantizer (``Callable[[Iterable[(name, tensor)]], Iterator[(name, tensor)]]``)."""

    def __init__(self, quantizer):
        self.quantizer = quantizer
        self._factors: dict[str, dict[str, Optional[tuple[int, ...]]]] = {}
        self._packs: dict[str, bool] = {}

    def factors(self, name: str, ndim: int) -> dict[str, Optional[tuple[int, ...]]]:
        """Output key -> block factor per dim (``None``: not shaped like the weight) of HF tensor
        ``name``; ``{}`` if the quantizer passes it through."""
        if name not in self._factors:
            probe = torch.ones((_PROBE,) * ndim, dtype=torch.bfloat16, device="cuda")
            module = _module(name)
            outs = [(n, t) for n, t in self.quantizer([(n, probe) for n in _group(name)]) if _module(n) == module]
            if any(t is probe for _n, t in outs):
                res = {}
            else:
                res = {
                    n[len(module) + 1 :]: tuple(_PROBE // s for s in t.shape)
                    if t.dim() == ndim and all(_PROBE % s == 0 for s in t.shape)
                    else None
                    for n, t in outs
                }
            self._factors[name] = res
        return self._factors[name]

    def aligned(self, name: str, global_shape, offset, local_shape) -> bool:
        """Whether the piece ``offset + local_shape`` of HF tensor ``name`` quantizes on its own."""
        meta = HFShardMeta(name, tuple(global_shape), torch.bfloat16, tuple(offset), tuple(local_shape))
        try:
            for f in self.factors(name, len(meta.global_shape)).values():
                _box(f, meta)
        except _Cut:
            return False
        return True

    def _packed(self, name: str) -> bool:
        """Whether the quantizer takes stacked expert tensor ``name`` in one call."""
        if name not in self._packs:
            probe = torch.ones(2, 256, 256, dtype=torch.bfloat16, device="cuda")
            try:
                list(self.quantizer([(name, probe)]))
                self._packs[name] = True
            except (NotImplementedError, ValueError):
                self._packs[name] = False
        return self._packs[name]

    def __call__(self, shards: Iterable[tuple[str, torch.Tensor, HFShardMeta]]) -> list:
        """Quantize ``(hf_name, local_view, meta)`` pieces into the rollout's tensors and their boxes."""
        out, plain = [], []
        for name, t, meta in shards:
            if stacked_kind(name) is not None and len(meta.global_shape) == 3:
                if self._packed(name):
                    out.extend(self._stack(name, t, meta))
                else:
                    plain.extend(_unstack_piece(name, t, meta))
            else:
                plain.append((name, t, meta))
        todo, feed = {}, []
        for name, t, meta in plain:
            if self.factors(name, t.dim()):
                todo[_module(name)] = (name, meta)
                feed.append((name, t))
            else:
                out.append((name, t, meta))
        for n, q in self.quantizer(feed):
            module = _module(n)
            src, meta = todo[module]
            out.append(_piece(n, q, meta, self._factors[src][n[len(module) + 1 :]]))
        return out

    def _stack(self, name: str, t: torch.Tensor, meta: HFShardMeta) -> list:
        prefix, kind, _key = stacked_kind(name)
        per = 2 if kind == "gate_up_proj" else 1
        rows: dict[str, list] = {}
        n_local = meta.local_shape[0] // per
        packed = t.reshape(n_local, per * meta.local_shape[1], meta.local_shape[2])
        for n, q in self.quantizer([(name, packed)]):
            m = _EXPERT_OUT.match(n)
            r = per * int(m["e"]) + (m["proj"] == "up_proj")
            rows.setdefault(m["key"], [None] * meta.local_shape[0])[r] = q
        row = HFShardMeta(
            expert_names(prefix, kind, 0)[0], meta.global_shape[1:], t.dtype, meta.offset[1:], meta.local_shape[1:]
        )
        factors = self.factors(row.name, 2)
        out = []
        for key, qs in rows.items():
            q = torch.stack(qs)
            per_row = tuple(q.shape[1:])
            g, off, size = _box(factors[key], row) or (per_row, (0,) * len(per_row), per_row)
            sname = f"{prefix}.mlp.experts.{kind}.{key}"
            meta_q = HFShardMeta(
                sname,
                (meta.global_shape[0], *g),
                q.dtype,
                (meta.offset[0], *off),
                (meta.local_shape[0], *size),
            )
            out.append((sname, q, _check(meta_q, q)))
        return out


def _unstack_piece(name: str, t: torch.Tensor, meta: HFShardMeta) -> list:
    """A local piece of a stacked expert tensor as per-expert pieces."""
    prefix, kind, _key = stacked_kind(name)
    per = 2 if kind == "gate_up_proj" else 1
    out = []
    for r in range(meta.local_shape[0]):
        e, j = divmod(meta.offset[0] + r, per)
        n = expert_names(prefix, kind, e)[j]
        out.append((n, t[r], HFShardMeta(n, meta.global_shape[1:], t.dtype, meta.offset[1:], meta.local_shape[1:])))
    return out


def _piece(name: str, q: torch.Tensor, meta: HFShardMeta, factors) -> tuple:
    q = q.contiguous()
    box = _box(factors, meta)
    if box is None:
        shape = tuple(q.shape)
        return name, q, HFShardMeta(name, shape, q.dtype, (0,) * len(shape), shape)
    g, off, size = box
    return name, q, _check(HFShardMeta(name, g, q.dtype, off, size), q)


def _check(meta: HFShardMeta, q: torch.Tensor) -> HFShardMeta:
    if tuple(q.shape) != tuple(meta.local_shape):
        raise RuntimeError(f"{meta.name}: quantized piece {tuple(q.shape)} does not match its box {meta.local_shape}")
    return meta
