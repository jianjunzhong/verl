# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Which part of each HF tensor this vLLM TP rank keeps, for ``rank_local_named_tensors``.

With it the ``sharded_nixl`` trainer sends a TP rank only its own box of a tensor. Instead of
per-model rules, a dry run of ``model.load_weights`` feeds each checkpoint name as a storage-less
fake tensor that only records ``narrow`` calls, and catches the loader's final ``copy_``. A name
whose loader copied exactly one box into exactly one parameter view gets a ``slices`` entry
(advertised in ``RolloutLayout.slices``) and a :class:`Scatter` (where a received box lands). The
box is the whole tensor for an expert an expert-parallel rank keeps, whose place is one row of a
FusedMoE weight; a name copied whole into a parameter of its own has nothing to place. Every other
name (several copies, more than 2 dims, any other op on the weight, loader error) is sent full and
goes through ``load_weights``.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class Scatter:
    """Where a received box goes: ``param.data.as_strided(shape, stride, param.storage_offset() + offset)``.

    ``pshape`` is the parameter's shape when the plan was baked; a parameter that no longer has it
    (a MoE layer staged out of its kernel layout) is left to the loader. ``None``: not checked.
    """

    param: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    pshape: tuple[int, ...] | None = None


@dataclass
class RankLocalPlan:
    slices: dict[str, dict] = field(default_factory=dict)
    scatters: dict[str, Scatter] = field(default_factory=dict)
    # (layer prefix, stacked kind, key) -> lowest expert this rank keeps, indexed on first use
    first_experts: dict[tuple[str, str, str], int] | None = None


class _Unsupported(RuntimeError):
    pass


class _BoxFake(torch.Tensor):
    """No storage; carries the box ``[(start, length), ...]`` of the full HF tensor it stands for."""

    @staticmethod
    def __new__(cls, box, device, sink):
        t = torch.Tensor._make_wrapper_subclass(cls, [n for _, n in box], dtype=torch.bfloat16, device=device)
        t._box, t._sink = list(box), sink
        return t

    def _narrow(self, dim, start, length):
        dim %= self.dim()
        s, n = self._box[dim]
        if start < 0 or start + length > n:
            raise _Unsupported(f"narrow({dim}, {start}, {length}) out of {n}")
        box = list(self._box)
        box[dim] = (s + start, length)
        return _BoxFake(box, self.device, self._sink)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.copy_:
            src = args[1] if len(args) > 1 else kwargs["src"]
            if isinstance(args[0], cls) or not isinstance(src, cls):
                raise _Unsupported("copy_ into a fake")
            src._sink((src._box, args[0]))
            return args[0]
        if func in (torch.Tensor.narrow, torch.narrow) and isinstance(args[0], cls):
            return args[0]._narrow(*args[1:], **kwargs)
        # metadata (shape, dim, numel, ...) works on the wrapper; any op that needs data reaches dispatch
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise _Unsupported(f"{func} on the weight")


def read_manifest(model_path: str) -> dict[str, list[int]]:
    """HF name -> full shape, from the safetensors headers only (no tensor data is read)."""
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        logger.warning("rank-local bake: no *.safetensors in %r, every tensor is sent full", model_path)
    manifest = {}
    for f in files:
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                manifest[k] = list(st.get_slice(k).get_shape())
    return manifest


def _dry_run(model: torch.nn.Module, entries: list, device) -> tuple[dict, int, Exception | None, bool]:
    """One ``model.load_weights`` call over ``entries`` (``[(name, full_shape), ...]``) as fakes.

    Returns the copies caught per name read, the index of the last entry read (``-1``: none), the
    exception the loader raised (``None`` if it finished) and whether a copy arrived for an entry
    other than the last one read (a loader that reads ahead: the index then need not be the culprit).
    """
    copies, at, ahead = {}, [-1], [False]

    def sink(i, caught):
        def put(c):
            ahead[0] |= i != at[0]
            caught.append(c)

        return put

    def stream():
        for i, (name, shape) in enumerate(entries):
            at[0] = i
            yield name, _BoxFake([(0, n) for n in shape], device, sink(i, copies.setdefault(name, [])))

    try:
        with torch.no_grad():
            model.load_weights(stream())
    except Exception as e:
        return copies, at[0], e, ahead[0]
    return copies, at[0], None, ahead[0]


def bake(model: torch.nn.Module, model_path: str) -> RankLocalPlan:
    """Dry-run ``model.load_weights`` over the checkpoint names; see the module docstring.

    All names go through one call, since every call rebuilds the model's name -> parameter tables
    (``O(#modules)`` per call: one call per name took 28 s on Qwen3-30B-A3B). A name the loader
    raises on is dropped and the call resumes after it; when the failing name cannot be told (the
    loader reads ahead), the names of that call are retried one call each.

    Runs on the real parameters: a loader may still write before it reaches the weight (e.g. zeroing
    the vocab padding rows), which is harmless because the trainer sends every weight right after.
    """
    device = next(model.parameters()).device
    owners = {p.untyped_storage().data_ptr(): (n, p) for n, p in model.named_parameters()}
    plan, errors = RankLocalPlan(), {}
    manifest = read_manifest(model_path)
    entries, copies = list(manifest.items()), {}
    while entries:
        got, at, err, ahead = _dry_run(model, entries, device)
        if err is None:
            copies.update(got)
            break
        if at < 0 or ahead:
            for entry in entries:
                got, _, err, _ = _dry_run(model, [entry], device)
                if err is None:
                    copies.update(got)
                else:
                    errors[entry[0]] = f"{type(err).__name__}: {err}"
            break
        # any loader we cannot follow keeps the full tensor
        errors[entries[at][0]] = f"{type(err).__name__}: {err}"
        got.pop(entries[at][0], None)
        copies.update(got)
        entries = entries[at + 1 :]
    for name, shape in manifest.items():
        caught = copies.get(name, [])
        if len(caught) != 1:  # 0: skipped by the model (e.g. tied lm_head); >1: several destinations
            continue
        box, dest = caught[0]
        lens = tuple(n for _, n in box)
        owner = owners.get(dest.untyped_storage().data_ptr())
        # the trainer's box planner cuts at most 2-D tensors (e.g. not stacked 3-D experts)
        if owner is None or tuple(dest.shape) != lens or len(shape) > 2:
            continue
        pname, p = owner
        # a tensor the loader copied whole still has a place when it is one slot of a larger
        # parameter: an expert-parallel rank keeps whole experts as rows of its FusedMoE weight,
        # and the stack of those rows is what the trainer cuts (see stacked_scatter)
        if list(lens) == shape and dest.numel() == p.numel():
            continue
        plan.slices[name] = {"full_shape": shape, "box": [list(b) for b in box]}
        plan.scatters[name] = Scatter(
            pname, dest.storage_offset() - p.storage_offset(), lens, tuple(dest.stride()), tuple(p.shape)
        )
    logger.info(
        "rank-local bake: %d names sliced, %d unsupported%s",
        len(plan.slices),
        len(errors),
        f" (e.g. {next(iter(errors.items()))})" if errors else "",
    )
    return plan


_EXPERT = re.compile(r"^(?P<prefix>.+)\.mlp\.experts\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<key>\w+)$")


def _first_experts(plan: RankLocalPlan) -> dict[tuple[str, str, str], int]:
    """Per layer and stacked kind, the lowest expert this rank has a place for."""
    if plan.first_experts is None:
        plan.first_experts = {}
        for n in plan.scatters:
            m = _EXPERT.match(n)
            if m is None:
                continue
            kind = "down_proj" if m["proj"] == "down_proj" else "gate_up_proj"
            k, e = (m["prefix"], kind, m["key"]), int(m["e"])
            plan.first_experts[k] = min(e, plan.first_experts.get(k, e))
    return plan.first_experts


def stacked_scatter(plan: RankLocalPlan, name: str, local_shape: tuple[int, ...]) -> Scatter | None:
    """The view a stacked expert box (``nccl_m2n_plan`` stacked names) of ``local_shape`` lands in,
    if the per-expert boxes of its rows tile one parameter contiguously in row order (FusedMoE's
    ``w13_weight`` / ``w2_weight``, whose rows are the whole model's experts without expert
    parallelism and this rank's run of them with it); ``None`` otherwise."""
    from verl.checkpoint_engine.nccl_m2n_plan import expert_names, stacked_kind

    sk = stacked_kind(name)
    if sk is None or len(local_shape) != 3:
        return None
    prefix, kind = sk
    per = 2 if kind == "gate_up_proj" else 1
    e0 = _first_experts(plan).get((prefix, kind, "weight"))
    if e0 is None:
        return None
    rows = [n for e in range(e0, e0 + local_shape[0] // per) for n in expert_names(prefix, kind, e)]
    scs = [plan.scatters.get(n) for n in rows]
    row_shape = tuple(local_shape[1:])
    row_stride = (row_shape[1], 1)
    step = row_shape[0] * row_shape[1]
    head = scs[0]
    for r, sc in enumerate(scs):
        if sc is None or sc.param != head.param or sc.shape != row_shape or sc.stride != row_stride:
            return None
        if sc.offset != head.offset + r * step:
            return None
    return Scatter(head.param, head.offset, tuple(local_shape), (step, *row_stride), head.pshape)


def apply(model: torch.nn.Module, plan: RankLocalPlan, weights):
    """Copy every received box straight into its parameter view; return the rest for ``load_weights``.

    Dispatches on shape: a tensor with its box's local shape is rank-local, a full-shape one (a
    ``named_tensors`` sync, or a name without a box) goes to the loader.
    """
    params, rest = dict(model.named_parameters()), []
    for name, t in weights:
        sc = plan.scatters.get(name)
        if sc is None or tuple(t.shape) != sc.shape:
            rest.append((name, t))
            continue
        p = params[sc.param]
        if sc.pshape is not None and tuple(p.shape) != sc.pshape:
            rest.append((name, t))
            continue
        p.data.as_strided(sc.shape, sc.stride, p.storage_offset() + sc.offset).copy_(t)
    return rest
