# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import os
from typing import Generator

import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine import CheckpointEngineRegistry, CheckpointEngineWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.workers.config import CheckpointEngineConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout import BaseRollout, RolloutReplica


class TrainingWorkerTest(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, checkpoint_engine_config: CheckpointEngineConfig) -> None:
        super().__init__(config)

        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        if torch.distributed.get_rank() == 0:
            engine_kwargs["is_master"] = True
        self.checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=bucket_size, **engine_kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        if getattr(self.checkpoint_engine, "send_weights_takes_engine", False):
            return await self.checkpoint_engine.send_weights(self.engine, global_steps=global_steps)
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


def mock_tp_slices(model_path: str, tp_size: int, tp_rank: int) -> dict:
    """vLLM-like TP boxes of a dense HF decoder: column-parallel weights split rows, row-parallel ones
    columns. ``layers.0`` k_proj gets the same box on every rank (like replicated KV heads), which
    is not a partition, so the engine must send it full."""
    import glob

    from safetensors import safe_open

    manifest = {}
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            manifest.update({k: list(st.get_slice(k).get_shape()) for k in st.keys()})
    col = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "embed_tokens")
    row = ("o_proj.weight", "down_proj.weight")
    slices = {}
    for name, shape in manifest.items():
        d = 0 if any(k in name for k in col) or name == "lm_head.weight" else 1 if name.endswith(row) else None
        if d is None or len(shape) != 2 or shape[d] % tp_size:
            continue
        box = [[0, n] for n in shape]
        step = shape[d] // tp_size
        box[d] = [0 if name.endswith("layers.0.self_attn.k_proj.weight") else tp_rank * step, step]
        slices[name] = {"full_shape": list(shape), "box": box}
    return slices


class MockServerAdapter(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        check_allclose: bool = True,
        shard_dst: bool = False,
    ):
        super().__init__(config, model_config, device_mesh=None)
        self.check_allclose = check_allclose
        self.shard_dst = shard_dst
        self.model = None
        self.received_weights: dict[str, torch.Tensor] = {}
        self.slices: dict = {}
        self.kill_after: int | None = None

    async def get_weight_ownership(self):
        from verl.workers.rollout.base import RolloutLayout

        tp = self.config.tensor_model_parallel_size
        tp_rank = int(os.environ["RANK"]) % tp
        if self.shard_dst and tp > 1:
            self.slices = mock_tp_slices(copy_to_local(self.model_config.path), tp, tp_rank)
        return RolloutLayout(
            tp_size=tp,
            tp_rank=tp_rank,
            wire_formats=["named_tensors", "rank_local_named_tensors"],
            slices=self.slices or None,
        )

    async def resume(self, tags: list[str]):
        raise NotImplementedError()

    async def release(self):
        raise NotImplementedError()

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        received = 0
        async for name, weight in weights:
            received += 1
            if self.kill_after is not None and received > self.kill_after:
                os._exit(1)  # simulate a rollout crash in the middle of a transfer
            weight = weight.clone()
            if self.check_allclose:
                self.received_weights[name] = weight.clone()

    def check_weights(self):
        if not self.check_allclose:
            return

        if self.model is None:
            local_path = copy_to_local(self.model_config.path)
            self.model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="cpu")

        boxes = 0
        for name, weight in self.model.state_dict().items():
            assert name in self.received_weights, f"weight {name} not received"
            received = self.received_weights[name]
            if tuple(received.shape) != tuple(weight.shape):
                entry = self.slices[name]
                weight = weight[tuple(slice(s, s + n) for s, n in entry["box"])]
                boxes += 1
            assert torch.allclose(weight.to(received.device, dtype=received.dtype), received), (
                f"weight {name} not equal"
            )
        self.received_weights.clear()
        print(f"Check passed, all weights are equal ({boxes} received as rank-local boxes)!")
        return boxes


class MockReplica(RolloutReplica):
    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        raise NotImplementedError

    async def launch_servers(self):
        """Launch http server in each node."""
        raise NotImplementedError


class CheckpointEngineWorkerTest(CheckpointEngineWorker):
    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        check_allclose: bool = True,
        shard_dst: bool = False,
        *args,
        **kwargs,
    ) -> None:
        server_adapter = MockServerAdapter(rollout_config, model_config, check_allclose, shard_dst)
        super().__init__(rollout_config, model_config, server_adapter, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_weights(self):
        return self.server_adapter.check_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_kill_after(self, num_tensors: int | None):
        self.server_adapter.kill_after = num_tensors


def create_trainer_worker_group(
    resource_pool: RayResourcePool,
    model_config: HFModelConfig,
    checkpoint_engine_config: CheckpointEngineConfig,
    strategy: str = "fsdp",
    fsdp_size: int | None = None,
) -> RayWorkerGroup:
    engine_config = FSDPEngineConfig(
        forward_only=True, fsdp_size=fsdp_size or resource_pool.world_size, strategy=strategy
    )
    trainer_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
    )

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(TrainingWorkerTest),
        config=trainer_config,
        checkpoint_engine_config=checkpoint_engine_config,
    )
    ray_cls_with_init.update_options(
        {
            "runtime_env": {
                "env_vars": {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            }
        }
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())
    return wg


async def create_rollout_worker_group(
    resource_pool: RayResourcePool,
    model_config: HFModelConfig,
    rollout_config: RolloutConfig,
    check_allclose: bool = True,
    shard_dst: bool = False,
) -> tuple[RayWorkerGroup, list[MockReplica]]:
    # create rollout worker group
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CheckpointEngineWorkerTest),
        model_config=model_config,
        rollout_config=rollout_config,
        check_allclose=check_allclose,
        shard_dst=shard_dst,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name())

    # create rollout replicas
    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    num_replicas = wg.world_size // rollout_world_size
    replicas = []
    for replica_rank in range(num_replicas):
        replica = MockReplica(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
        )
        replicas.append(replica)
    await asyncio.gather(*[replica.init_hybrid(wg) for replica in replicas])

    return wg, replicas
