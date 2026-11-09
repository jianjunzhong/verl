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
"""Megatron HF shard source vs Megatron-Bridge's gather export, bitwise.

Every rank exports its local HF shards through
:class:`verl.workers.engine.megatron.hf_shard_source.MegatronHFShardSource` (probe +
cached runs, no communication) and checks each one against the same box of the full
tensor from ``bridge.export_hf_weights``. Rank 0 also checks that the shipped boxes
cover every HF tensor exactly once per pipeline stage, and that the forced gather
fallback yields the full tensor.

    TP_SIZE=2 torchrun --nproc_per_node=2 tests/special_distributed/test_megatron_hf_shard_source.py
    TP_SIZE=2 PP_SIZE=2 torchrun --nproc_per_node=4 tests/special_distributed/test_megatron_hf_shard_source.py
    TP_SIZE=4 HF_CKPT=/path/Qwen3-8B torchrun --nproc_per_node=4 \
        tests/special_distributed/test_megatron_hf_shard_source.py

``HF_CKPT`` unset builds a tiny Qwen3 (uneven-free shapes, untied embeddings).
"""

import os
import pathlib
import sys
import time

import torch
import torch.distributed as dist

TP = int(os.environ.get("TP_SIZE", "2"))
PP = int(os.environ.get("PP_SIZE", "1"))
HF_CKPT = os.environ.get("HF_CKPT")


def _tiny_qwen3(rank: int) -> str:
    from transformers import AutoConfig

    path = "/tmp/tiny_qwen3_hf_shard_source"
    if rank == 0 and not pathlib.Path(path, "model.safetensors").exists():
        from transformers import AutoModelForCausalLM

        cfg = AutoConfig.for_model(
            "qwen3",
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=1024,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        cfg.architectures = ["Qwen3ForCausalLM"]
        AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    dist.barrier()
    return path


class _Engine:
    """The two attributes MegatronHFShardSource reads from a MegatronEngine."""

    def __init__(self, bridge, model):
        from verl.workers.engine.megatron.delta_export import build_export_index

        self._delta_slot_cache: dict = {}
        self._index = build_export_index(bridge, model, self._delta_slot_cache)

    def _mcore_export_index(self):
        return self._index


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))

    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    mpu.initialize_model_parallel(tensor_model_parallel_size=TP, pipeline_model_parallel_size=PP)
    model_parallel_cuda_manual_seed(1234)
    ckpt = HF_CKPT or _tiny_qwen3(rank)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(ckpt, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = PP
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    model = [m.cuda().to(torch.bfloat16) for m in (model if isinstance(model, list) else [model])]
    bridge.load_hf_weights(model)

    from verl.workers.engine.megatron.hf_shard_source import MegatronHFShardSource

    engine = _Engine(bridge, model)
    force = {"model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight"}
    source = MegatronHFShardSource(engine, force_fallback=force)
    t0 = time.perf_counter()
    source.plan()
    t_plan = time.perf_counter() - t0
    shards = [(n, v.clone(), m) for n, v, m in source.iter_shards()]

    ref = {n: t.detach().to(torch.bfloat16) for n, t in bridge.export_hf_weights(model, cpu=False, show_progress=False)}
    if bridge.hf_pretrained.config.tie_word_embeddings:
        # Bridge re-emits the tied embedding as lm_head; mcore has no output layer to ship and
        # the rollout ties lm_head to embed_tokens itself
        ref.pop("lm_head.weight", None)

    failures, counts = [], {"R": 0, "S0": 0, "S1": 0}
    for name, view, meta in shards:
        full = ref.get(name)
        if full is None:
            failures.append(f"{name}: not in bridge export")
            continue
        if tuple(full.shape) != meta.global_shape:
            failures.append(f"{name}: shape {meta.global_shape} != {tuple(full.shape)}")
            continue
        box = full[tuple(slice(o, o + n) for o, n in zip(meta.offset, meta.local_shape, strict=True))]
        if not torch.equal(box.view(torch.int16), view.view(torch.int16)):
            failures.append(f"{name}: bitwise mismatch")
        if name in force and not meta.is_full:
            failures.append(f"{name}: forced fallback did not yield the full tensor")
        d = meta.shard_dim()
        counts["R" if d is None else f"S{d}"] += 1

    # coverage: within each pp stage, the dp-rank-0 ranks' boxes tile every HF tensor
    mine = [(n, m.offset, m.local_shape) for n, _, m in shards]
    every: list = [None] * world
    dist.all_gather_object(every, (mpu.get_pipeline_model_parallel_rank(), mine))
    if rank == 0:
        covered: dict = {}
        for _pp, boxes in every:
            for n, off, size in boxes:
                covered.setdefault(n, set()).add((off, size))
        for n, full in ref.items():
            area = sum(torch.Size(size).numel() for _, size in covered.get(n, ()))
            if area < full.numel():
                failures.append(f"{n}: boxes cover {area} of {full.numel()} elements")

    # steady-state cost: cached runs, no communication
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        for _n, _v, _m in source.iter_shards():
            pass
    torch.cuda.synchronize()
    t_steady = (time.perf_counter() - t0) / 3

    all_fail: list = [None] * world
    dist.all_gather_object(all_fail, failures)
    if rank == 0:
        flat = [f for fs in all_fail for f in fs]
        for f in flat[:20]:
            print("FAIL:", f)
        print(
            f"HF SHARD SOURCE [tp={TP} pp={PP} ckpt={os.path.basename(ckpt)}]: rank0 {len(shards)} shards {counts}, "
            f"plan {t_plan:.2f}s, steady {t_steady * 1e3:.1f} ms, fallback rows {source.fallback_names()}"
        )
        print("HF SHARD SOURCE PASSED" if not flat else "HF SHARD SOURCE FAILED")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and any(all_fail):
        sys.exit(1)


if __name__ == "__main__":
    main()
