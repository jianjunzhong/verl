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
"""End-to-end ``nccl_m2n``: FSDP2 trainer local shards -> rollout replicas, checked against the HF checkpoint.

MODEL_PATH=~/models/Qwen/Qwen3-8B pytest -s tests/checkpoint_engine/test_nccl_m2n_on_gpu.py
"""

import os

import pytest
import ray

from tests.checkpoint_engine.test_utils import create_rollout_worker_group, create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.device import get_torch_device
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig

_ngpus = get_torch_device().device_count()
pytest.importorskip("nccl.m2n")


@pytest.mark.asyncio
@pytest.mark.parametrize("shard_dst", [False, True])
@pytest.mark.parametrize("group_bytes", [0, 64 << 20])
@pytest.mark.parametrize("num_trainer, rollout_tp", [(2, 2)])
@auto_await
async def test_nccl_m2n_fsdp(shard_dst, group_bytes, num_trainer, rollout_tp, bucket_size_mb=512):
    model_path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/models/Qwen/Qwen3-8B"))
    num_rollout = _ngpus - num_trainer
    ray.init(runtime_env={"env_vars": {"VERL_LOGGING_LEVEL": "INFO"}})

    checkpoint_engine_config = CheckpointEngineConfig(
        backend="nccl_m2n",
        update_weights_bucket_megabytes=bucket_size_mb,
        engine_kwargs={"nccl_m2n": {"group_bytes": group_bytes}},
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(
        name="vllm", tensor_model_parallel_size=rollout_tp, checkpoint_engine=checkpoint_engine_config
    )

    resource_pool = RayResourcePool(process_on_nodes=[_ngpus], max_colocate_count=3)
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config, strategy="fsdp2")
    actor_wg.reset()
    rollout, replicas = await create_rollout_worker_group(
        rollout_pool, model_config, rollout_config, check_allclose=True, shard_dst=shard_dst
    )
    manager = CheckpointEngineManager(config=checkpoint_engine_config, actor_wg=actor_wg, replicas=replicas)

    for step in range(3):
        metrics = await manager.update_weights(global_steps=step)
        boxes = rollout.check_weights()
        print(f"step {step}: {metrics}")
        assert (min(boxes) > 0) == shard_dst, boxes
        if step > 0:
            # the epoch (plan, communicators) is built once per membership
            assert not any(v for k, v in metrics.items() if k.endswith("epoch_build_s")), metrics

    ray.shutdown()
