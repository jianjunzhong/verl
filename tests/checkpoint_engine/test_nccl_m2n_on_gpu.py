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

Every check clears what a rollout rank received, so a replica that missed an update fails its next check.
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


async def _setup(num_trainer, rollout_tp, engine_kwargs, shard_dst, bucket_size_mb=512):
    model_path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/models/Qwen/Qwen3-8B"))
    ray.init(runtime_env={"env_vars": {"VERL_LOGGING_LEVEL": "INFO"}})
    checkpoint_engine_config = CheckpointEngineConfig(
        backend="nccl_m2n",
        update_weights_bucket_megabytes=bucket_size_mb,
        engine_kwargs={"nccl_m2n": engine_kwargs},
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(
        name="vllm", tensor_model_parallel_size=rollout_tp, checkpoint_engine=checkpoint_engine_config
    )
    resource_pool = RayResourcePool(process_on_nodes=[_ngpus], max_colocate_count=3)
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, _ngpus - num_trainer])
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config, strategy="fsdp2")
    actor_wg.reset()
    rollout, replicas = await create_rollout_worker_group(
        rollout_pool, model_config, rollout_config, check_allclose=True, shard_dst=shard_dst
    )
    return checkpoint_engine_config, actor_wg, rollout, replicas


def _check(replicas):
    return [b for r in replicas for b in ray.get([w.check_weights.remote() for w in r.workers])]


def _state(replica):
    return ray.get(replica.workers[0].execute_checkpoint_engine.remote("channel_state"))


@pytest.mark.asyncio
@pytest.mark.parametrize("shard_dst", [False, True])
@pytest.mark.parametrize("group_bytes", [0, 64 << 20])
@pytest.mark.parametrize("num_trainer, rollout_tp", [(2, 2)])
@auto_await
async def test_nccl_m2n_fsdp(shard_dst, group_bytes, num_trainer, rollout_tp):
    config, actor_wg, rollout, replicas = await _setup(num_trainer, rollout_tp, {"group_bytes": group_bytes}, shard_dst)
    manager = CheckpointEngineManager(config=config, actor_wg=actor_wg, replicas=replicas)

    for step in range(3):
        metrics = await manager.update_weights(global_steps=step)
        boxes = rollout.check_weights()
        print(f"step {step}: {metrics}")
        assert (min(boxes) > 0) == shard_dst, boxes
        # channels are opened once per replica
        assert metrics["nccl_m2n/opened"] == (len(replicas) if step == 0 else 0), metrics

    ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_channels", [32, 2])
@auto_await
async def test_nccl_m2n_membership(max_channels, num_trainer=2, rollout_tp=2):
    """Join, leave and rejoin without touching the other replicas; a small channel budget forces
    symmetric generation resets, including for a replica that was parked across one."""
    config, actor_wg, _rollout, replicas = await _setup(
        num_trainer, rollout_tp, {"max_channels": max_channels}, shard_dst=True
    )
    assert len(replicas) == 3, "needs 2 trainer GPUs + 3 TP2 rollout replicas"
    a, b, c = replicas
    manager = CheckpointEngineManager(config=config, actor_wg=actor_wg, replicas=[a, b])

    async def sync(step, expect):
        metrics = await manager.update_weights(global_steps=step)
        print(f"step {step} {[replicas.index(r) for r in manager.replicas]}: {metrics}")
        for k, v in expect.items():
            assert metrics[f"nccl_m2n/{k}"] == v, (k, metrics)
        assert min(_check(manager.replicas)) > 0

    await sync(0, {"opened": 2, "left": 0, "reset": 0})
    a0, b0 = _state(a), _state(b)
    if max_channels >= 3:
        manager.add_replicas([c])
        await sync(1, {"opened": 1, "left": 0, "reset": 0})
        assert _state(a) == a0, "a join must not touch other replicas"
        manager.remove_replicas([b])
        await sync(2, {"opened": 0, "left": 1, "rejoined": 0})
        manager.add_replicas([b])
        await sync(3, {"opened": 0, "left": 0, "rejoined": 1, "reset": 0})
        assert _state(b) == b0, "a rejoining replica reuses its parked communicator"
        assert _state(a) == a0
    else:
        manager.remove_replicas([b])
        await sync(1, {"opened": 0, "left": 1})
        # a + b known, c would be the third channel: new generation, a and c rebuilt
        manager.add_replicas([c])
        await sync(2, {"opened": 2, "reset": 1})
        assert _state(a)["gen"] == 1 and _state(a)["opened"] == a0["opened"] + 1
        manager.remove_replicas([a])
        await sync(3, {"left": 1})
        # b is still on generation 0 with a communicator the trainer aborted
        manager.add_replicas([b])
        await sync(4, {"opened": 2, "reset": 1})
        assert _state(b)["gen"] == 2 and _state(c)["gen"] == 2
    ray.shutdown()


@pytest.mark.asyncio
@auto_await
async def test_nccl_m2n_kill_mid_transfer(num_trainer=2, rollout_tp=2):
    """A rollout rank dying mid-transfer: the other replicas still get complete weights, the sync
    fails, generation stays fenced, and the trainer sender raises at its deadline and refuses
    further transfers."""
    # small buckets: the receiver posts two buckets ahead, so the victim must die with more left
    config, actor_wg, _rollout, replicas = await _setup(
        num_trainer, rollout_tp, {"sync_timeout_s": 30}, True, bucket_size_mb=16
    )
    manager = CheckpointEngineManager(config=config, actor_wg=actor_wg, replicas=replicas)
    await manager.update_weights(global_steps=0)
    assert min(_check(replicas)) > 0

    victim = replicas[-1]
    ray.get(victim.workers[1].set_kill_after.remote(5))
    with pytest.raises(ray.exceptions.RayActorError):
        await manager.update_weights(global_steps=1)
    assert manager._weights_dirty
    with pytest.raises(RuntimeError, match="mixed"):
        await manager.resume_generation_replicas()

    with pytest.raises(ray.exceptions.RayTaskError, match="did not complete"):
        ray.get(manager._update_ticket.trainer_published)
    assert min(_check(replicas[:-1])) > 0, "healthy replicas must receive complete weights"
    with pytest.raises(ray.exceptions.RayTaskError, match="unusable"):
        ray.get(actor_wg.update_weights(global_steps=2))
    ray.shutdown()
