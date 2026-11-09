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

from verl.checkpoint_engine.nccl_m2n_plan import Piece, Side, build_channel_plan, plan_digest, split_buckets


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


def test_split_buckets():
    man = _manifests(("a", (4, 4), None), ("b", (4, 4), None), ("c", (64, 4), None), ("d", (2,), None), n=1)
    plan = build_channel_plan(man, [None])  # bytes: a 32, b 32, c 512, d 4
    buckets = split_buckets(plan, bucket_bytes=64)
    assert [[[it.name for it in g] for g in b] for b in buckets] == [[["a"], ["b"]], [["c"]], [["d"]]]
    grouped = split_buckets(plan, bucket_bytes=64, group_bytes=64)
    assert [[[it.name for it in g] for g in b] for b in grouped] == [[["a", "b"]], [["c"]], [["d"]]]
