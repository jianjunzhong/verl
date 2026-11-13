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
"""``nccl_m2n`` into real vLLM workers (``update_weights_from_m2n``): FSDP2 trainers -> offline
``vllm.LLM`` TP replicas, no rollout server. Every vLLM parameter is zeroed first and must be
bit-exact again after the sync; most of them are written by M2N in place.

MODEL_PATH=~/models/Qwen/Qwen3-0.6B pytest -s tests/checkpoint_engine/test_nccl_m2n_vllm_on_gpu.py
"""

import os
import pickle

import pytest
import ray

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.utils.device import get_torch_device
from verl.workers.config import CheckpointEngineConfig, HFModelConfig

pytest.importorskip("nccl.m2n")
pytest.importorskip("vllm")

_ngpus = get_torch_device().device_count()


def _snapshot_and_zero(self):
    model = self.model_runner.model
    self._test_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.data.zero_()


def _mismatches(self):
    import torch

    model = self.model_runner.model
    return [n for n, p in model.named_parameters() if not torch.equal(p, self._test_before[n])]


def _expert_boxes(self):
    """How many of this worker's boxes are expert weights (0 without a rank-local plan)."""
    plan = self._get_rank_local_plan()
    return sum("experts" in n for n in ((plan.slices if plan else None) or {}))


def _replica_check(self):
    """``(mismatching parameters, expert boxes)`` of every rank of the replica, gathered inside it.

    vLLM returns only the first engine's results from a ``collective_rpc``, so a DP rank other
    than 0 would go unchecked.
    """
    import torch
    from vllm.distributed import parallel_state as ps

    mine = (_mismatches(self), _expert_boxes(self))
    group = ps.get_ep_group() if _ep_on(self) else ps.get_tp_group()
    out = [None] * group.world_size
    torch.distributed.all_gather_object(out, mine, group=group.cpu_group)
    return out


def _ep_on(self) -> bool:
    return bool(getattr(self.model_runner.vllm_config.parallel_config, "enable_expert_parallel", False))


@ray.remote(num_gpus=2)
class _VLLMReplica:
    def __init__(
        self,
        model_path: str,
        tp: int,
        skip_tokenizer_init: bool = False,
        max_model_len: int = 1024,
        expert_parallel: bool = False,
    ):
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # collective_rpc with test functions
        from vllm import LLM

        self.llm = LLM(
            model=model_path,
            skip_tokenizer_init=skip_tokenizer_init,
            tensor_parallel_size=tp,
            enable_expert_parallel=expert_parallel,
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=max_model_len,
            distributed_executor_backend="mp",
            worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        )

    def actor_id(self):
        return ray.get_runtime_context().get_actor_id()

    def rpc(self, method, *args, **kwargs):
        return self.llm.collective_rpc(method, args=args, kwargs=kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("group_bytes", [0])
@pytest.mark.parametrize("moe, ep", [(False, False), (True, False), (True, True)])
async def test_nccl_m2n_into_vllm(group_bytes, moe, ep, num_trainer=2, rollout_tp=2, num_replicas=2, bucket_size_mb=64):
    """``moe``: a tiny Qwen3-MoE, whose stacked transformers-v5 experts (FSDP2 shards on the expert
    dim) land straight in FusedMoE's ``w13_weight`` / ``w2_weight``. ``ep``: each rank keeps a run
    of whole experts instead of a slice of every expert, so the stack is cut on the expert dim."""
    from tests.checkpoint_engine.test_nccl_m2n_on_gpu import _tiny_moe
    from verl.checkpoint_engine.nccl_m2n_checkpoint_engine import NCCLM2NCheckpointEngine, _Client
    from verl.single_controller.ray.base import RayResourcePool

    if num_trainer + rollout_tp * num_replicas > _ngpus:
        pytest.skip("not enough GPUs")
    tokenizer_path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/models/Qwen/Qwen3-0.6B"))
    # tied: FSDP2 meta-device init of an untied model fails with transformers 5.5 + accelerate 1.12
    model_path = _tiny_moe(tied=True) if moe else tokenizer_path
    # vLLM's NVLS communicators and the M2N windows exhaust the NVSwitch multicast pool otherwise
    ray.init(runtime_env={"env_vars": {"VERL_LOGGING_LEVEL": "INFO", "NCCL_NVLS_ENABLE": "0"}})
    config = CheckpointEngineConfig(
        backend="nccl_m2n",
        update_weights_bucket_megabytes=bucket_size_mb,
        engine_kwargs={"nccl_m2n": {"group_bytes": group_bytes}},
    )
    # the borrowed tokenizer's special ids are beyond the tiny model's vocab
    override = {k: 0 for k in ("bos_token_id", "eos_token_id", "pad_token_id")} if moe else {}
    model_config = HFModelConfig(
        path=model_path, tokenizer_path=tokenizer_path, use_remove_padding=True, override_config=override
    )
    trainer_pool = RayResourcePool(process_on_nodes=[num_trainer], max_colocate_count=3)
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, config, strategy="fsdp2")
    actor_wg.reset()
    replicas = [
        _VLLMReplica.remote(
            model_path,
            rollout_tp,
            skip_tokenizer_init=moe,
            max_model_len=256 if moe else 1024,
            expert_parallel=ep,
        )
        for _ in range(num_replicas)
    ]
    layouts = ray.get([r.rpc.remote("get_weight_ownership") for r in replicas])
    for lay in layouts:
        assert [x["tp_rank"] for x in lay] == list(range(rollout_tp))
        assert all("nccl_m2n" in x["wire_formats"] and x.get("slices") for x in lay), lay[0].keys()
    if moe:
        # EP keeps whole experts, TP a slice of every one: either way every expert has a place
        assert all(n > 0 for n in ray.get(replicas[0].rpc.remote(_expert_boxes)))
    ids = ray.get([r.actor_id.remote() for r in replicas])

    payload = {"bucket_size": bucket_size_mb << 20, "group_bytes": group_bytes, "timeout_s": 300.0}
    client = None
    for step in range(3):
        ray.get([r.rpc.remote(_snapshot_and_zero) for r in replicas])
        trainer_meta = ray.get(actor_wg.execute_checkpoint_engine(["prepare"] * num_trainer))
        rollout_meta = [
            {"role": "rollout", "actor_id": f"{ids[i]}-{t}", "layout": lay[t]}
            for i, lay in enumerate(layouts)
            for t in range(rollout_tp)
        ]
        actor_kwargs, rollout_kwargs = NCCLM2NCheckpointEngine.build_topology(
            num_trainer, len(rollout_meta), trainer_meta + rollout_meta
        )
        ray.get(actor_wg.execute_checkpoint_engine(method=["init_process_group"] * num_trainer, **actor_kwargs))
        client = client or _Client(actor_kwargs["master"][0], 300.0)
        sent = actor_wg.update_weights(global_steps=step)
        received = []
        update = rollout_kwargs["update"][0]
        for i, r in enumerate(replicas):
            reply = client.request(("plan", update, rollout_kwargs["replica"][i * rollout_tp], None))
            # what CheckpointEngineWorker.receive_weights forwards: the reply and the replica layout
            update_info = {
                **payload,
                "reply": pickle.dumps(reply),
                "slices": rollout_kwargs["slices"][i * rollout_tp],
            }
            received.append(r.rpc.remote("update_weights_from_m2n", update_info=update_info))
        metrics = ray.get(sent)
        results = ray.get(received)
        print(f"step {step}: trainer {metrics[0]}, rollout {results[0][0]}")
        for res in results:
            # most tensors are written in place, the rest are loaded through load_weights
            assert all(x["direct"] > x["loaded"] > 0 for x in res), res
        assert ray.get([r.rpc.remote(_mismatches) for r in replicas]) == [[[]] * rollout_tp] * num_replicas

    ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("rollout_dp", [1, 2])
async def test_nccl_m2n_vllm_server(rollout_dp, num_trainer=2, rollout_tp=2, num_replicas=2):
    """The full path: CheckpointEngineManager -> CheckpointEngineWorker -> vLLM ServerAdapter ->
    ``update_weights_from_m2n`` in the workers of a vLLM server, then generation.

    ``rollout_dp`` > 1 runs one MoE replica over a DP x TP mesh with expert parallelism (vLLM
    wants ``ep == tp * dp``), the layout the rollout of a large MoE is deployed in."""
    from hydra import compose, initialize_config_dir

    from tests.checkpoint_engine.test_nccl_m2n_on_gpu import _tiny_moe
    from verl.checkpoint_engine import CheckpointEngineManager
    from verl.single_controller.ray import RayResourcePool
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.tokenizer import normalize_token_ids
    from verl.workers.rollout.llm_server import LLMServerManager

    moe = rollout_dp > 1
    if moe:
        num_replicas = 1
    if num_trainer + rollout_tp * rollout_dp * num_replicas > _ngpus:
        pytest.skip("not enough GPUs")
    tokenizer_path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/models/Qwen/Qwen3-0.6B"))
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")
    config.actor_rollout_ref.model.path = _tiny_moe(tied=True) if moe else tokenizer_path
    config.actor_rollout_ref.model.tokenizer_path = tokenizer_path
    if moe:
        # the borrowed tokenizer's special ids are beyond the tiny model's vocab
        config.actor_rollout_ref.model.override_config = {
            k: 0 for k in ("bos_token_id", "eos_token_id", "pad_token_id")
        }
    rollout = config.actor_rollout_ref.rollout
    rollout.name = "vllm"
    rollout.tensor_model_parallel_size = rollout_tp
    rollout.data_parallel_size = rollout_dp
    rollout.expert_parallel_size = rollout_tp * rollout_dp if moe else 1
    rollout.nnodes, rollout.n_gpus_per_node = 1, rollout_tp * rollout_dp * num_replicas
    rollout.gpu_memory_utilization = 0.3
    rollout.enforce_eager = True
    if moe:
        rollout.max_model_len, rollout.prompt_length, rollout.response_length = 256, 64, 32
    rollout.load_format = "auto"  # the snapshot to compare against must be the real weights
    rollout.checkpoint_engine.backend = "nccl_m2n"
    rollout.checkpoint_engine.update_weights_bucket_megabytes = 64
    config.trainer.n_gpus_per_node, config.trainer.nnodes = num_trainer, 1

    ray.init(
        runtime_env={
            "env_vars": {
                "VERL_LOGGING_LEVEL": "INFO",
                "NCCL_NVLS_ENABLE": "0",
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",  # collective_rpc with test functions
            }
        }
    )
    model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    ckpt_config = omega_conf_to_dataclass(rollout.checkpoint_engine)
    trainer_pool = RayResourcePool(process_on_nodes=[num_trainer], max_colocate_count=3)
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, ckpt_config, strategy="fsdp2")
    actor_wg.reset()
    llm_server_manager = await LLMServerManager.create(config=config)
    replicas = llm_server_manager.get_replicas()
    checkpoint_manager = CheckpointEngineManager(config=ckpt_config, actor_wg=actor_wg, replicas=replicas)

    servers = [s for r in replicas for s in r.servers]
    if moe:  # the tiny model's vocab is far below the borrowed tokenizer's ids
        prompt_ids = [1, 2, 3, 4]
    else:
        prompt = [{"role": "user", "content": "How to make a sandwich?"}]
        prompt_ids = normalize_token_ids(
            model_config.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        )
    client = llm_server_manager.get_client()
    for step in range(2):
        ray.get([s.collective_rpc.remote(_snapshot_and_zero) for s in servers])
        await checkpoint_manager.update_weights(global_steps=step)
        for results in ray.get([s.collective_rpc.remote(_replica_check) for s in servers]):
            # every rank of the replica: all weights back, and a place for its experts under EP
            replica = results[0]
            assert len(replica) == rollout_tp * rollout_dp, replica
            assert all(bad == [] and (boxes > 0) == moe for bad, boxes in replica), replica
        output = await client.generate(
            request_id=f"test_{step}", prompt_ids=prompt_ids, sampling_params={"temperature": 0.0, "max_tokens": 32}
        )
        assert output.extra_fields["global_steps"] == step
        if not moe:
            print(f"step {step}: {model_config.tokenizer.decode(output.token_ids, skip_special_tokens=True)!r}")
        assert len(output.token_ids) > 0

    ray.shutdown()
