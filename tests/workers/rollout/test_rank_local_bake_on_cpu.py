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

import pytest
import torch

from verl.workers.rollout.vllm_rollout import rank_local_bake
from verl.workers.rollout.vllm_rollout.rank_local_bake import Scatter, apply, bake

MANIFEST = {"a": [8, 3], "b": [4, 3], "c": [4, 3], "norm": [3], "tied": [8, 3]}


class _Toy(torch.nn.Module):
    """TP rank 1 of 2: ``a`` row-split into ``w``; ``b`` / ``c`` row-split into halves of ``fused``."""

    def __init__(self, bad=(), read_ahead=False):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
        self.fused = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
        self.norm = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
        self.bad, self.read_ahead, self.calls = set(bad), read_ahead, 0

    def load_weights(self, weights):
        self.calls += 1
        if self.read_ahead:
            weights = list(weights)
        for name, w in weights:
            if name in self.bad:
                w = w.t()
            if name == "a":
                self.w.data.copy_(w.narrow(0, 4, 4))
            elif name in ("b", "c"):
                self.fused.data.narrow(0, 0 if name == "b" else 2, 2).copy_(w.narrow(0, 2, 2))
            elif name == "norm":
                self.norm.data.copy_(w)


@pytest.fixture(autouse=True)
def _manifest(monkeypatch):
    monkeypatch.setattr(rank_local_bake, "read_manifest", lambda path: dict(MANIFEST))


def test_bake_one_load_weights_call():
    model = _Toy()
    plan = bake(model, "unused")
    assert model.calls == 1
    assert plan.slices == {
        "a": {"full_shape": [8, 3], "box": [[4, 4], [0, 3]]},
        "b": {"full_shape": [4, 3], "box": [[2, 2], [0, 3]]},
        "c": {"full_shape": [4, 3], "box": [[2, 2], [0, 3]]},
    }
    assert plan.scatters["a"] == Scatter("w", 0, (4, 3), (3, 1))
    assert plan.scatters["c"] == Scatter("fused", 6, (2, 3), (3, 1))

    full = {n: torch.arange(torch.Size(s).numel(), dtype=torch.float32).reshape(s) for n, s in MANIFEST.items()}
    rest = apply(model, plan, [(n, full[n][2:4]) for n in ("b", "c")] + [("norm", full["norm"])])
    assert [n for n, _ in rest] == ["norm"]
    assert torch.equal(model.fused, torch.cat([full["b"][2:4], full["c"][2:4]]))


@pytest.mark.parametrize("read_ahead", [False, True])
def test_bake_drops_only_the_name_the_loader_raises_on(read_ahead):
    model = _Toy(bad={"b"}, read_ahead=read_ahead)
    plan = bake(model, "unused")
    assert set(plan.slices) == {"a", "c"}
    assert plan.scatters["c"] == Scatter("fused", 6, (2, 3), (3, 1))
    # resumed after "b" in a second call; a loader reading ahead is retried one call per name
    assert model.calls == (1 + len(MANIFEST) if read_ahead else 2)
