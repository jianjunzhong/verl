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
import pytest
import torch

from verl.checkpoint_engine.nccl_m2n_plan import (
    Piece,
    PlanItem,
    Side,
    build_channel_plan,
    plan_digest,
    split_buckets,
    unstack,
)


def _pieces(name, shape, n, dim):
    """``n`` senders holding an even split of ``shape`` on ``dim`` (``None``: all full)."""
    out = []
    for k in range(n):
        off, size = [0] * len(shape), list(shape)
        if dim is not None:
            size[dim] //= n
            off[dim] = k * size[dim]
        out.append(Piece(name, tuple(shape), "bfloat16", tuple(off), tuple(size)))
    return out


def _manifests(*tensors, n=2):
    per = [[] for _ in range(n)]
    for name, shape, dim in tensors:
        for k, p in enumerate(_pieces(name, shape, n, dim)):
            per[k].append(p)
    return per


def _slices(name, shape, n, dim):
    out = []
    for k in range(n):
        box = [[0, s] for s in shape]
        box[dim] = [k * shape[dim] // n, shape[dim] // n]
        out.append({name: {"full_shape": list(shape), "box": box}})
    return out


def test_source_placements_and_replicated_destination():
    man = _manifests(("w", (8, 4), 0), ("norm", (4,), None), ("o", (4, 8), 1))
    plan = build_channel_plan(man, [None, None, None])
    by = {it.name: it for it in plan}
    assert [it.name for it in plan] == ["w", "norm", "o"]
    assert by["w"].src == Side(0, 2, 0, (4, 4))
    assert by["norm"].src == Side(0, 2, None, (4,))
    assert by["o"].src == Side(0, 2, 1, (4, 4))
    for it in plan:
        assert it.dst == Side(2, 3, None, it.global_shape)


def test_sharded_destination_needs_every_rank_to_agree():
    man = _manifests(("q", (8, 4), 0), ("o", (4, 8), 1))
    q, o = _slices("q", (8, 4), 2, 0), _slices("o", (4, 8), 2, 1)
    plan = build_channel_plan(man, [{**q[0], **o[0]}, {**q[1], **o[1]}])
    assert plan[0].dst == Side(2, 2, 0, (4, 4))
    assert plan[1].dst == Side(2, 2, 1, (4, 4))

    # replicated KV heads: both ranks keep the same rows -> not a partition -> full
    same = [{"q": q[0]["q"]}, {"q": q[0]["q"]}]
    assert build_channel_plan(man, same)[0].dst == Side(2, 2, None, (8, 4))
    # one rank without a box -> full
    assert build_channel_plan(man, [q[0], None])[0].dst.shard_dim is None
    # uneven split (vLLM vocab padding) -> full
    uneven = [{"q": {"full_shape": [8, 4], "box": b}} for b in ([[0, 5], [0, 4]], [[5, 3], [0, 4]])]
    assert build_channel_plan(man, uneven)[0].dst.shard_dim is None
    # layout of another checkpoint shape -> full
    other = [{"q": {"full_shape": [16, 4], "box": b}} for b in ([[0, 8], [0, 4]], [[8, 8], [0, 4]])]
    assert build_channel_plan(man, other)[0].dst.shard_dim is None


def test_data_parallel_destination_is_a_replicated_mesh():
    """A DP x TP replica: every DP rank asks for the same TP boxes, so the boxes repeat."""
    man = _manifests(("q", (8, 4), 0))
    tp2 = _slices("q", (8, 4), 2, 0)
    dst = build_channel_plan(man, tp2 + tp2)[0].dst
    assert dst == Side(2, 4, 0, (4, 4), repl=2)
    assert dst.shard_n == 2
    # the boxes alone say how the ranks repeat: TP4 over one DP rank is a 4-way split
    tp4 = _slices("q", (8, 4), 4, 0)
    assert build_channel_plan(man, tp4)[0].dst == Side(2, 4, 0, (2, 4), repl=1)
    # a replicated side ignores repl: every rank takes the whole tensor
    assert build_channel_plan(man, tp2 + tp2[::-1])[0].dst == Side(2, 4, None, (8, 4))


def test_expert_parallel_destination_cuts_the_stack_on_the_expert_dim():
    E, inter, hidden, pre = 8, 6, 4, "model.layers.0"
    gu, dn = f"{pre}.mlp.experts.gate_up_proj", f"{pre}.mlp.experts.down_proj"
    man = _manifests((gu, (2 * E, inter, hidden), 0), (dn, (E, hidden, inter), 0))

    def ep_slices(ep):
        """Each rank keeps whole experts: the boxes of its run, the rest of the layer missing."""
        out = []
        for r in range(ep):
            sl = {}
            for e in range(r * E // ep, (r + 1) * E // ep):
                base = f"{pre}.mlp.experts.{e}."
                for p in ("gate_proj", "up_proj"):
                    sl[base + p + ".weight"] = {"full_shape": [inter, hidden], "box": [[0, inter], [0, hidden]]}
                sl[base + "down_proj.weight"] = {"full_shape": [hidden, inter], "box": [[0, hidden], [0, inter]]}
            out.append(sl)
        return out

    by = {it.name: it for it in build_channel_plan(man, ep_slices(4))}
    assert by[gu].dst == Side(2, 4, 0, (2 * E // 4, inter, hidden), repl=1)
    assert by[dn].dst == Side(2, 4, 0, (E // 4, hidden, inter), repl=1)
    # rank 2 of 4 receives experts 4 and 5
    t = torch.arange(4 * inter * hidden).view(4, inter, hidden)
    assert [n for n, _ in unstack(by[gu], t, rank=2)] == [
        f"{pre}.mlp.experts.{e}.{p}.weight" for e in (4, 5) for p in ("gate_proj", "up_proj")
    ]

    # a rank holding half an expert (an uneven split): the stack goes full
    uneven = ep_slices(4)
    moved = f"{pre}.mlp.experts.2.gate_proj.weight"
    uneven[0][moved] = uneven[1].pop(moved)
    assert build_channel_plan(man, uneven)[0].dst.shard_dim is None


def test_subset_holders_and_errors():
    # tensor held by senders 1..2 only (e.g. a later stage's piece), sharded over them
    a = _pieces("x", (4, 2), 2, 0)
    man = [[Piece("y", (2,), "bfloat16", (0,), (2,))], [a[0]], [a[1]]]
    plan = build_channel_plan(man, [None])
    assert plan[1].src == Side(1, 2, 0, (2, 2))
    assert plan[0].src == Side(0, 1, None, (2,))

    with pytest.raises(ValueError, match="contiguous"):
        build_channel_plan([[a[0]], [], [a[1]]], [None])
    uneven = [Piece("z", (5, 2), "bfloat16", (0, 0), (3, 2)), Piece("z", (5, 2), "bfloat16", (3, 0), (2, 2))]
    with pytest.raises(ValueError, match="gather"):
        build_channel_plan([[uneven[0]], [uneven[1]]], [None])
    with pytest.raises(ValueError, match="1-D to 3-D"):
        build_channel_plan([[Piece("t", (1, 1, 1, 1), "bfloat16", (0,) * 4, (1,) * 4)]], [None])
    big = _manifests(("w", (32, 4), 0), n=32)
    with pytest.raises(ValueError, match="16"):
        build_channel_plan(big, [None])


def test_digest_is_deterministic_and_layout_sensitive():
    man = _manifests(("q", (8, 4), 0))
    q = _slices("q", (8, 4), 2, 0)
    a = plan_digest(build_channel_plan(man, [None, None]))
    assert a == plan_digest(build_channel_plan(man, [None, None]))
    assert a != plan_digest(build_channel_plan(man, q))


def _expert_slices(prefix, n_e, inter, hidden, tp):
    """Per-expert TP slices as vLLM's FusedMoE loader keeps them: gate/up on rows, down on cols."""
    out = []
    for k in range(tp):
        s, n = k * inter // tp, inter // tp
        sl = {}
        for e in range(n_e):
            base = f"{prefix}.mlp.experts.{e}."
            for p in ("gate_proj", "up_proj"):
                sl[base + p + ".weight"] = {"full_shape": [inter, hidden], "box": [[s, n], [0, hidden]]}
            sl[base + "down_proj.weight"] = {"full_shape": [hidden, inter], "box": [[0, hidden], [s, n]]}
        out.append(sl)
    return out


def test_stacked_experts_take_the_per_expert_destination_boxes():
    E, inter, hidden, pre = 4, 8, 6, "model.layers.0"
    gu, dn = f"{pre}.mlp.experts.gate_up_proj", f"{pre}.mlp.experts.down_proj"
    # EP=2 senders: each holds 2 experts of both stacks
    man = _manifests((gu, (2 * E, inter, hidden), 0), (dn, (E, hidden, inter), 0))
    plan = build_channel_plan(man, _expert_slices(pre, E, inter, hidden, 2))
    by = {it.name: it for it in plan}
    assert by[gu].src == Side(0, 2, 0, (E, inter, hidden))
    assert by[gu].dst == Side(2, 2, 1, (2 * E, inter // 2, hidden))
    assert by[dn].dst == Side(2, 2, 2, (E, hidden, inter // 2))

    t = torch.arange(2 * E * 4 * hidden).view(2 * E, 4, hidden)
    names = [n for n, _ in unstack(by[gu], t, rank=1)]
    assert names[:3] == [f"{pre}.mlp.experts.0.gate_proj.weight", f"{pre}.mlp.experts.0.up_proj.weight"] + [
        f"{pre}.mlp.experts.1.gate_proj.weight"
    ]
    assert torch.equal(unstack(by[gu], t, rank=1)[3][1], t[3])
    assert unstack(PlanItem("x", (2,), "bfloat16", by[gu].src, by[gu].dst), t, 0) == [("x", t)]

    # one expert sliced differently: the stack goes full
    odd = _expert_slices(pre, E, inter, hidden, 2)
    odd[0][f"{pre}.mlp.experts.3.down_proj.weight"]["box"] = [[0, hidden], [0, inter]]
    by = {it.name: it for it in build_channel_plan(man, odd)}
    assert by[dn].dst.shard_dim is None and by[gu].dst.shard_dim == 1


def test_stacked_scatter_needs_contiguous_expert_rows():
    from verl.workers.rollout.vllm_rollout.rank_local_bake import RankLocalPlan, Scatter, stacked_scatter

    E, rows, hidden, pre = 3, 4, 6, "model.layers.0"
    gu = f"{pre}.mlp.experts.gate_up_proj"
    plan = RankLocalPlan()
    for e in range(E):
        for j, p in enumerate(("gate_proj", "up_proj")):
            off = (2 * e + j) * rows * hidden
            plan.scatters[f"{pre}.mlp.experts.{e}.{p}.weight"] = Scatter("w13", off, (rows, hidden), (hidden, 1))
    sc = stacked_scatter(plan, gu, (2 * E, rows, hidden))
    assert sc == Scatter("w13", 0, (2 * E, rows, hidden), (rows * hidden, hidden, 1))

    # expert parallelism: the rows this rank has a place for are experts 2..4 of the model
    local = RankLocalPlan()
    for e in range(2, 2 + E):
        for j, p in enumerate(("gate_proj", "up_proj")):
            off = (2 * (e - 2) + j) * rows * hidden
            local.scatters[f"{pre}.mlp.experts.{e}.{p}.weight"] = Scatter("w13", off, (rows, hidden), (hidden, 1))
    assert stacked_scatter(local, gu, (2 * E, rows, hidden)) == sc
    # up before gate (a swapped kernel layout): no single view
    a, b = (f"{pre}.mlp.experts.1.{p}.weight" for p in ("gate_proj", "up_proj"))
    plan.scatters[a], plan.scatters[b] = plan.scatters[b], plan.scatters[a]
    assert stacked_scatter(plan, gu, (2 * E, rows, hidden)) is None
    assert stacked_scatter(plan, "model.layers.0.mlp.gate.weight", (2 * E, rows, hidden)) is None


def test_fsdp_restack_gate_up():
    from verl.workers.engine.fsdp.hf_shard_source import _restack
    from verl.workers.engine.hf_shard import HFShardMeta

    name = "model.layers.1.mlp.experts.gate_up_proj"
    m = _restack(HFShardMeta(name, (8, 16, 6), torch.bfloat16, (4, 0, 0), (4, 16, 6)))
    assert (m.global_shape, m.offset, m.local_shape) == ((16, 8, 6), (8, 0, 0), (8, 8, 6))
    down = HFShardMeta("model.layers.1.mlp.experts.down_proj", (8, 6, 8), torch.bfloat16, (0, 0, 0), (8, 6, 8))
    assert _restack(down) is down
    with pytest.raises(ValueError, match="gate/up"):
        _restack(HFShardMeta(name, (8, 16, 6), torch.bfloat16, (0, 8, 0), (8, 8, 6)))


def test_split_buckets():
    man = _manifests(("a", (4, 4), None), ("b", (4, 4), None), ("c", (64, 4), None), ("d", (2,), None), n=1)
    plan = build_channel_plan(man, [None])  # bytes: a 32, b 32, c 512, d 4
    buckets = split_buckets(plan, bucket_bytes=64)
    assert [[[it.name for it in g] for g in b] for b in buckets] == [[["a"], ["b"]], [["c"]], [["d"]]]
    grouped = split_buckets(plan, bucket_bytes=64, group_bytes=64)
    assert [[[it.name for it in g] for g in b] for b in grouped] == [[["a", "b"]], [["c"]], [["d"]]]
