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
"""CPU tests for the hooks shared by point-to-point checkpoint engines (nccl_m2n, sharded_nixl)."""

import pytest
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineManager
from verl.workers.engine.hf_shard import HFShardMeta, dense_box
from verl.workers.engine.megatron.delta_export import _take_runs
from verl.workers.engine.utils import hf_shard_export
from verl.workers.rollout.base import RolloutLayout


def test_engine_hook_defaults():
    assert CheckpointEngine.send_weights_takes_engine is False
    assert CheckpointEngine.needs_rollout_layout is False


def test_rollout_layout_round_trip():
    layout = RolloutLayout(
        tp_size=4,
        tp_rank=1,
        layer_range=(0, 18),
        local_experts={3: [4, 5]},
        wire_formats=["named_tensors", "rank_local_named_tensors"],
        slices={"a": {"full_shape": [8, 4], "box": [[2, 2], [0, 4]]}},
    )
    d = layout.to_dict()
    # Ray / JSON turn tuples into lists and int keys into str keys
    d["layer_range"] = list(d["layer_range"])
    d["local_experts"] = {str(k): v for k, v in d["local_experts"].items()}
    assert RolloutLayout.from_dict(d) == layout
    assert RolloutLayout.from_dict(None) is None


def test_take_runs():
    x = torch.arange(10)
    assert torch.equal(_take_runs(x, [(2, 3)]), torch.tensor([2, 3, 4]))
    assert _take_runs(x, [(2, 3)]).data_ptr() == x[2:].data_ptr()  # one run: a view, no copy
    assert torch.equal(_take_runs(x, [(7, 2), (0, 2)]), torch.tensor([7, 8, 0, 1]))


@pytest.mark.parametrize(
    "shape,offset,size",
    [((8, 4), (0, 0), (8, 4)), ((8, 4), (2, 0), (2, 4)), ((8, 4), (0, 1), (8, 2)), ((6, 4, 3), (0, 2, 0), (6, 2, 3))],
)
def test_dense_box(shape, offset, size):
    full = torch.arange(torch.Size(shape).numel()).view(shape)
    pos = full[tuple(slice(o, o + n) for o, n in zip(offset, size, strict=True))].reshape(-1)
    assert dense_box(pos, shape) == (offset, size)


def test_dense_box_rejects_non_boxes():
    full = torch.arange(32).view(8, 4)
    assert dense_box(torch.cat([full[0], full[2]]), (8, 4)) is None  # gap
    assert dense_box(full[0:2].reshape(-1).flip(0), (8, 4)) is None  # not ascending
    assert dense_box(torch.empty(0, dtype=torch.int64), (8, 4)) is None


def test_hf_shard_meta_shard_dim():
    assert HFShardMeta("w", (8, 4), torch.bfloat16, (0, 0), (8, 4)).shard_dim() is None
    assert HFShardMeta("w", (8, 4), torch.bfloat16, (4, 0), (4, 4)).shard_dim() == 0
    assert HFShardMeta("w", (8, 4), torch.bfloat16, (0, 2), (8, 2)).shard_dim() == 1
    with pytest.raises(ValueError):
        HFShardMeta("w", (8, 4), torch.bfloat16, (0, 0), (4, 2)).shard_dim()


def test_hf_shard_export_with_and_without_indices():
    from verl.workers.engine.spec import ShardSpec

    seen = []

    def entry(name, spec, place, lidx, lval):
        seen.append((name, None if lidx is None else lidx.tolist()))
        return [(name, (lval.numel(),))], "bfloat16", torch.tensor([lval.numel()]), lidx, lval

    shard = torch.arange(4, dtype=torch.bfloat16)
    # ``contributes`` is read only together with an explicit ``place`` override
    replica = ShardSpec(full_shape=(4,), place=0, contributes=False)
    gen = [("a", shard, ShardSpec(full_shape=(4,))), ("b", shard, replica)]
    out = list(hf_shard_export(iter(gen), entry, with_indices=False))
    assert seen[0] == ("a", None)
    assert seen[1] == ("b", [])  # non-contributing rank keeps lockstep with an empty entry
    assert out[0][-1] is None  # unsharded: no gather group
    seen.clear()
    list(hf_shard_export(iter(gen[:1]), entry, with_indices=True))
    assert seen == [("a", [0, 1, 2, 3])]


def test_manager_blocks_resume_while_weights_dirty():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.replicas = []
    manager._weights_dirty = True
    manager._applied_global_steps = 3
    with pytest.raises(RuntimeError, match="global step 3"):
        manager.resume_generation_replicas()
    manager._weights_dirty = False
    manager.resume_generation_replicas()
