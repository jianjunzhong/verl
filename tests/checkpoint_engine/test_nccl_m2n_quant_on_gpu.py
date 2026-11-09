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
"""``LocalShardQuantizer``: quantizing every trainer rank's piece and placing each output at its box
is bitwise the quantization of the full tensor. One GPU, no communication.

pytest -s tests/checkpoint_engine/test_nccl_m2n_quant_on_gpu.py
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("compressed_tensors")

from verl.checkpoint_engine.nccl_m2n_plan import PlanItem, Side, stacked_kind, unstack  # noqa: E402
from verl.workers.engine.hf_shard import HFShardMeta  # noqa: E402
from verl.workers.engine.hf_shard_quant import LocalShardQuantizer  # noqa: E402

# (linear, moe) presets; MXFP4 / NVFP4 are routed-expert only
SCHEMES = [("FP8_BLOCK", "FP8_BLOCK"), ("MXFP8", "MXFP8"), ("MXFP8", "MXFP4"), (None, "NVFP4")]
Q_PROJ = "model.layers.0.self_attn.q_proj.weight"
PREFIX = "model.layers.0"


def _quantizer(linear, moe):
    from verl.utils.qat import dispatch_rollout_weight_quantizer
    from verl.workers.config.rollout import RolloutQuantizationConfig

    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_moe"), local_path=None)
    rollout = SimpleNamespace(name="vllm", quantization=RolloutQuantizationConfig(linear=linear, moe=moe))
    return dispatch_rollout_weight_quantizer(model_config=model_config, rollout_config=rollout)


def _pieces(name, full, dim, n):
    """``n`` even pieces of ``full`` on ``dim`` (``None``: one full piece), as ``(name, view, meta)``."""
    out = []
    for k in range(1 if dim is None else n):
        off, size = [0] * full.dim(), list(full.shape)
        if dim is not None:
            size[dim] //= n
            off[dim] = k * size[dim]
        view = full[tuple(slice(o, o + s) for o, s in zip(off, size, strict=True))].contiguous()
        out.append((name, view, HFShardMeta(name, tuple(full.shape), full.dtype, tuple(off), tuple(size))))
    return out


def _bytes(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _assemble(outputs):
    """``name -> full tensor`` from every rank's ``(name, tensor, meta)`` outputs; stacked experts are
    unstacked as a receiver does."""
    full = {}
    for name, t, m in outputs:
        assert tuple(t.shape) == tuple(m.local_shape) and t.dtype == m.dtype, (name, t.shape, m)
        buf = full.setdefault(name, torch.zeros(m.global_shape, dtype=_bytes(t).dtype, device=t.device))
        buf[tuple(slice(o, o + s) for o, s in zip(m.offset, m.local_shape, strict=True))] = _bytes(t)
    out = {}
    for name, t in full.items():
        side = Side(0, 1, None, tuple(t.shape))
        out.update(unstack(PlanItem(name, tuple(t.shape), "uint8", side, side), t, 0))
    return out


def _check(got, ref):
    ref = {n: _bytes(t) for n, t in ref}
    assert sorted(got) == sorted(ref)
    for n, t in ref.items():
        assert torch.equal(got[n], t), n


@pytest.mark.parametrize("linear, moe", SCHEMES)
@pytest.mark.parametrize("dim", [None, 0, 1])
def test_dense_pieces(linear, moe, dim):
    quantizer = _quantizer(linear, moe)
    q = LocalShardQuantizer(quantizer)
    w = torch.randn(512, 1024, dtype=torch.bfloat16, device="cuda")
    pieces = _pieces(Q_PROJ, w, dim, 4)  # 128 rows / 256 cols
    assert all(q.aligned(n, m.global_shape, m.offset, m.local_shape) for n, _v, m in pieces)
    got = _assemble([o for p in pieces for o in q([p])])
    _check(got, quantizer([(Q_PROJ, w)]))


def _experts(n_e, inter, hidden):
    torch.manual_seed(0)
    per = {}
    shapes = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
    for e in range(n_e):
        for proj, shape in shapes.items():
            per[f"{PREFIX}.mlp.experts.{e}.{proj}.weight"] = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    gate_up = torch.stack(
        [per[f"{PREFIX}.mlp.experts.{e}.{p}.weight"] for e in range(n_e) for p in ("gate_proj", "up_proj")]
    )
    down = torch.stack([per[f"{PREFIX}.mlp.experts.{e}.down_proj.weight"] for e in range(n_e)])
    return per, gate_up, down


@pytest.mark.parametrize("linear, moe", SCHEMES)
@pytest.mark.parametrize("dims", [(0, 0), (1, 2)])  # EP2 on the expert dim; ETP2 on I
def test_stacked_expert_pieces(linear, moe, dims):
    quantizer = _quantizer(linear, moe)
    q = LocalShardQuantizer(quantizer)
    per, gate_up, down = _experts(4, 256, 512)
    pieces = _pieces(f"{PREFIX}.mlp.experts.gate_up_proj", gate_up, dims[0], 2)
    pieces += _pieces(f"{PREFIX}.mlp.experts.down_proj", down, dims[1], 2)
    if moe == "NVFP4" and dims != (0, 0):
        # gate and up share a global scale over the whole expert: an ETP cut cannot be quantized alone
        name = f"{PREFIX}.mlp.experts.0.gate_proj.weight"
        assert not q.aligned(name, (256, 512), (128, 0), (128, 512))
        return
    outputs = [o for p in pieces for o in q([p])]
    if moe != "NVFP4":
        # one stacked tensor per output, not one per expert
        assert all(stacked_kind(n) for n, _t, _m in outputs), [n for n, _t, _m in outputs]
    _check(_assemble(outputs), quantizer(per.items()))


def test_cut_blocks_are_not_aligned():
    fp8 = LocalShardQuantizer(_quantizer("FP8_BLOCK", "FP8_BLOCK"))
    mx = LocalShardQuantizer(_quantizer("MXFP8", "MXFP8"))
    # 192-row pieces cut 128-row FP8 blocks; MXFP8 groups 1 x 32
    assert not fp8.aligned(Q_PROJ, (384, 1024), (192, 0), (192, 1024))
    assert mx.aligned(Q_PROJ, (384, 1024), (192, 0), (192, 1024))
    assert not mx.aligned(Q_PROJ, (128, 96), (0, 48), (128, 48))
    # the end of the tensor may hold a partial block
    assert fp8.aligned(Q_PROJ, (320, 1024), (256, 0), (64, 1024))
    # pass-through names are always local
    assert fp8.aligned("model.norm.weight", (1024,), (512,), (512,))
