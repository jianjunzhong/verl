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
"""``nccl_m2n``: trainer local shards -> rollout shards with NCCL M2N reshard (nccl-extensions).

Each trainer rank sends the HF-coordinate pieces it holds (``engine.get_hf_shard_source()``, no
gather) and each rollout rank receives only its own box of every tensor (from
``RolloutLayout.slices``; tensors without a box arrive full). See :mod:`.nccl_m2n_plan` for how
the per-tensor layouts are derived.

Lifecycle:

* ``prepare`` / ``build_topology`` / ``init_process_group`` run on every update but only record the
  membership (rollout replicas and their layouts). Nothing is built or torn down while it does
  not change.
* The first send/receive of a new membership builds the epoch: trainer ranks all-gather their
  manifests, rank 0 publishes them with one communicator unique id per (trainer PP stage, rollout
  replica) channel over a ZMQ REP socket, every rank derives the same plan and checks its digest,
  and each channel gets its own communicator and stream.
* Steady state: one ``Handle.reshard`` per tensor straight from the source view, stream-ordered,
  no staging window, no per-weight host sync and no per-weight metadata. Senders wait for one
  event per channel; receivers fill double-buffered buckets and yield them.
* A membership change resets symmetrically: every rank destroys its handle before its
  communicators (nccl-extensions 0.1.0 caches communicator state by address while a handle lives).

M2N cannot be aborted once a transfer has started (a dead peer hangs every survivor of its
communicator), so a sync that misses ``sync_timeout_s`` raises and the processes of that channel
must be restarted.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import threading
import time
from contextlib import nullcontext
from typing import Any, AsyncGenerator, Optional

import ray
import torch
import torch.distributed as dist
import zmq

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_m2n_plan import Piece, PlanItem, Side, build_channel_plan, plan_digest, split_buckets
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

try:
    import nccl.m2n as m2n
    from nccl.core import Communicator, UniqueId, get_unique_id
    from nccl.m2n import DistTensor, Handle, Mesh, Replicate, Shard
except ImportError as e:  # pragma: no cover - depends on nccl-extensions
    raise ImportError("checkpoint_engine.backend=nccl_m2n needs nccl-extensions (pip install nccl-extensions)") from e

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["NCCLM2NCheckpointEngine"]

# every bucket view starts on this byte alignment
_ALIGN = 256


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


class _Server:
    """Trainer rank 0's control plane: answers ``("epoch", key)`` with the epoch info once it is
    built and ``("go", key, step)`` once the trainer has started sending that step (any step if
    ``step`` is None)."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.port, _ = get_free_port(self.ip)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        if is_valid_ipv6_address(self.ip):
            self._sock.setsockopt(zmq.IPV6, 1)
            self._sock.bind(f"tcp://[{self.ip}]:{self.port}")
        else:
            self._sock.bind(f"tcp://{self.ip}:{self.port}")
        self._cv = threading.Condition()
        self._epochs: dict[str, dict] = {}
        self._go: Optional[tuple[str, Any]] = None
        self._thread = threading.Thread(target=self._serve, name="nccl_m2n_ctl", daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> dict:
        return {"ip": self.ip, "port": self.port}

    def publish_epoch(self, key: str, info: dict) -> None:
        with self._cv:
            self._epochs = {key: info}
            self._cv.notify_all()

    def publish_go(self, key: str, step) -> None:
        with self._cv:
            self._go = (key, step)
            self._cv.notify_all()

    def _serve(self) -> None:
        while True:
            try:
                req = self._sock.recv_pyobj()
            except zmq.ZMQError:
                return
            kind, key = req[0], req[1]
            with self._cv:
                if kind == "epoch":
                    self._cv.wait_for(lambda k=key: k in self._epochs, timeout=self.timeout_s)
                    rep = self._epochs.get(key)
                else:
                    want = (key, req[2]) if req[2] is not None else None

                    def ready(k=key, want=want):
                        return self._go is not None and self._go[0] == k and (want is None or self._go == want)

                    rep = self._go if self._cv.wait_for(ready, timeout=self.timeout_s) else None
            self._sock.send(pickle.dumps(rep))


class _Client:
    def __init__(self, endpoint: dict, timeout_s: float):
        self.timeout_s = timeout_s
        ip, port = endpoint["ip"], endpoint["port"]
        self._addr = f"tcp://[{ip}]:{port}" if is_valid_ipv6_address(ip) else f"tcp://{ip}:{port}"
        self._ipv6 = is_valid_ipv6_address(ip)
        self._ctx = zmq.Context()
        self._sock = None

    def request(self, req: tuple):
        if self._sock is None:
            self._sock = self._ctx.socket(zmq.REQ)
            self._sock.setsockopt(zmq.LINGER, 0)
            if self._ipv6:
                self._sock.setsockopt(zmq.IPV6, 1)
            self._sock.connect(self._addr)
        self._sock.send_pyobj(req)
        if not self._sock.poll(int(self.timeout_s * 1000)):
            self._sock.close()
            self._sock = None
            raise TimeoutError(f"nccl_m2n: trainer control plane did not answer {req[0]!r} in {self.timeout_s}s")
        rep = self._sock.recv_pyobj()
        if rep is None:
            raise TimeoutError(f"nccl_m2n: trainer has not reached {req!r} in {self.timeout_s}s")
        return rep


def _desc(item: PlanItem, side: Side, buf: Optional[torch.Tensor]) -> DistTensor:
    if side.shard_dim is None:
        mesh, placements = Mesh((side.n, 1), start_rank=side.start), [Replicate(), Replicate()]
    else:
        mesh, placements = Mesh((1, side.n), start_rank=side.start), [Replicate(), Shard(side.shard_dim)]
    return DistTensor(buf, local_shape=side.local_shape, dtype=_dtype(item.dtype), mesh=mesh, placements=placements)


@CheckpointEngineRegistry.register("nccl_m2n")
class NCCLM2NCheckpointEngine(CheckpointEngine):
    """NCCL M2N checkpoint engine.

    Args:
        bucket_size: receive bucket in bytes; rollout ranks hold two of them.
        is_master: trainer rank 0 (serves the control plane).
        group_bytes: batch consecutive reshards of up to this many destination bytes into one
            ``m2n.group()`` (0: one call per tensor). Must be equal on every rank. On a single
            NVLink node batching is slower in most cases; on EFA it may be required.
        sync_timeout_s: deadline for the control plane and for one sync's transfers.
    """

    send_weights_takes_engine = True
    needs_rollout_layout = True

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        group_bytes: int = 0,
        sync_timeout_s: float = 600.0,
    ) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.group_bytes = int(group_bytes)
        self.sync_timeout_s = float(sync_timeout_s)

        self._is_rollout = False
        self._layout = None
        self._server = _Server(self.sync_timeout_s) if is_master else None
        self._client: Optional[_Client] = None

        # registered by init_process_group
        self._membership: Optional[str] = None
        self._replicas_slices: list[list[Optional[dict]]] = []
        self._replica: Optional[int] = None
        self._replica_rank: Optional[int] = None

        # built for one membership epoch
        self._epoch: Optional[str] = None
        self._handle = None
        self._comms: dict[tuple[int, int], Any] = {}
        self._streams: dict[tuple[int, int], torch.cuda.Stream] = {}
        self._buckets: dict[tuple[int, int], list] = {}
        self._recv_bufs: list[torch.Tensor] = []

    # ------------------------------------------------------------------ topology

    def set_rollout_layout(self, layout) -> None:
        self._is_rollout = True
        self._layout = layout

    def prepare(self) -> dict:
        if self._is_rollout:
            return {
                "role": "rollout",
                "actor_id": ray.get_runtime_context().get_actor_id(),
                "layout": None if self._layout is None else self._layout.to_dict(),
            }
        return {"role": "trainer", "master": self._server.endpoint if self._server else None}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        master = metadata[0]["master"]
        assert master is not None, "trainer rank 0 must be the nccl_m2n master"
        rollout = metadata[actor_wg_world_size:]

        replicas: list[list[int]] = []
        for i, meta in enumerate(rollout):
            layout = meta["layout"]
            if layout is None:
                raise NotImplementedError(
                    "nccl_m2n needs every rollout worker to report its RolloutLayout (get_weight_ownership)"
                )
            if layout["pp_size"] != 1:
                raise NotImplementedError("nccl_m2n does not support rollout pipeline parallelism yet")
            if layout["tp_rank"] == 0:
                replicas.append([])
            if not replicas or len(replicas[-1]) != layout["tp_rank"]:
                raise ValueError(f"nccl_m2n: rollout worker {i} is out of TP order ({layout['tp_rank']})")
            replicas[-1].append(i)
        for rep in replicas:
            if len(rep) != rollout[rep[0]]["layout"]["tp_size"]:
                raise ValueError(f"nccl_m2n: incomplete rollout replica {rep}")

        slices = [[rollout[i]["layout"].get("slices") for i in rep] for rep in replicas]
        ident = [[rollout[i]["actor_id"] for i in rep] for rep in replicas]
        membership = hashlib.sha256(pickle.dumps((ident, slices))).hexdigest()[:16]

        actor_kwargs = {
            "role": ["trainer"] * actor_wg_world_size,
            "master": [master] * actor_wg_world_size,
            "membership": [membership] * actor_wg_world_size,
            "replicas_slices": [slices] * actor_wg_world_size,
        }
        replica_of = [None] * rollout_world_size
        for r, rep in enumerate(replicas):
            for t, i in enumerate(rep):
                replica_of[i] = (r, t)
        rollout_kwargs = {
            "role": ["rollout"] * rollout_world_size,
            "master": [master] * rollout_world_size,
            "membership": [membership] * rollout_world_size,
            "replicas_slices": [[slices[r]] for r, _ in replica_of],
            "replica": [r for r, _ in replica_of],
            "replica_rank": [t for _, t in replica_of],
        }
        return actor_kwargs, rollout_kwargs

    def init_process_group(
        self,
        role: str,
        master: dict,
        membership: str,
        replicas_slices: list,
        replica: Optional[int] = None,
        replica_rank: Optional[int] = None,
    ):
        """Record the membership; communicators are (re)built on the first transfer of a new one."""
        if role == "rollout" and self._client is None:
            self._client = _Client(master, self.sync_timeout_s)
        self._membership = membership
        self._replicas_slices = replicas_slices
        self._replica, self._replica_rank = replica, replica_rank

    def finalize(self):
        """Keep communicators and buffers: they live as long as the membership."""

    def _reset_epoch(self) -> None:
        if self._handle is None and not self._comms:
            return
        torch.cuda.synchronize()
        if self._handle is not None:
            self._handle.destroy()
            self._handle = None
        for comm in self._comms.values():
            comm.destroy()
        self._comms, self._streams, self._buckets = {}, {}, {}
        self._epoch = None

    def _init_channels(self, uids: dict, sizes: dict, my_rank: dict) -> None:
        """Create the handle and this rank's channel communicators, in global (stage, replica)
        order on every rank so overlapping communicator inits cannot deadlock."""
        self._handle = Handle.create()
        for key in sorted(my_rank):
            self._comms[key] = Communicator.init(sizes[key], my_rank[key], UniqueId.from_bytes(uids[key]))
            self._streams[key] = torch.cuda.Stream()

    # ------------------------------------------------------------------ trainer

    async def _build_trainer_epoch(self, source, shards) -> None:
        stage, sender = getattr(source, "stage", 0), source.sender_index
        pieces = []
        if sender is not None:
            pieces = [
                Piece(n, tuple(m.global_shape), _dtype_name(v.dtype), tuple(m.offset), tuple(m.local_shape))
                for n, v, m in shards
            ]
        gathered: list = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (stage, sender, pieces))
        by_stage: dict[int, dict[int, list]] = {}
        for st, s, pcs in gathered:
            if s is not None:
                assert s not in by_stage.setdefault(st, {}), f"stage {st} has two senders with index {s}"
                by_stage[st][s] = pcs
        manifests = {}
        for st, d in sorted(by_stage.items()):
            assert sorted(d) == list(range(len(d))), f"stage {st} senders {sorted(d)} are not 0..n-1"
            manifests[st] = [d[i] for i in range(len(d))]

        R = len(self._replicas_slices)
        box = [None]
        if self.is_master:
            box = [{(st, r): bytes(get_unique_id()) for st in manifests for r in range(R)}]
        dist.broadcast_object_list(box, src=0)
        uids = box[0]

        plans = {
            (st, r): build_channel_plan(manifests[st], self._replicas_slices[r]) for st in manifests for r in range(R)
        }
        digests = {k: plan_digest(v) for k, v in plans.items()}
        self._reset_epoch()
        if self._server is not None:
            self._server.publish_epoch(self._membership, {"manifests": manifests, "uids": uids, "digests": digests})
        if sender is not None:
            mine = {k: sender for k in plans if k[0] == stage}
            sizes = {k: len(manifests[k[0]]) + len(self._replicas_slices[k[1]]) for k in mine}
            self._init_channels(uids, sizes, mine)
            self._buckets = {k: split_buckets(plans[k], self.bucket_size, self.group_bytes) for k in mine}
        self._epoch = self._membership
        n_fallback = sum(1 for _n, _v, m in shards if m.is_full and sender is not None)
        logger.info(
            "nccl_m2n epoch %s: stage %s sender %s, %d channels, %d tensors (%d full), digests %s",
            self._membership,
            stage,
            sender,
            len(self._comms),
            len(shards),
            n_fallback,
            sorted(set(digests.values())),
        )

    @torch.no_grad()
    async def send_weights(self, engine, global_steps: int | None = None):
        """Send this rank's local HF shards of ``engine`` to every rollout replica."""
        t0 = time.perf_counter()
        source = engine.get_hf_shard_source()
        shards = list(source.iter_shards())
        t_export = time.perf_counter()
        built = self._epoch != self._membership
        if built:
            await self._build_trainer_epoch(source, shards)
        if self._server is not None:
            self._server.publish_go(self._membership, global_steps)
        t_plan = time.perf_counter()
        metrics = {"nccl_m2n/export_s": t_export - t0, "nccl_m2n/epoch_build_s": t_plan - t_export if built else 0.0}
        if not self._comms:
            return metrics

        local = {n: v for n, v, _m in shards}
        cur = torch.cuda.current_stream()
        for s in self._streams.values():
            s.wait_stream(cur)
        keys = sorted(self._buckets)
        for i in range(max(len(b) for b in self._buckets.values())):
            for key in keys:
                if i < len(self._buckets[key]):
                    for group in self._buckets[key][i]:
                        self._enqueue(key, group, local, send=True)
        t_enq = time.perf_counter()
        events = [s.record_event() for s in self._streams.values()]
        await self._wait(events, "send")
        metrics.update({"nccl_m2n/enqueue_s": t_enq - t_plan, "nccl_m2n/send_s": time.perf_counter() - t_plan})
        return metrics

    def _enqueue(self, key, group: list[PlanItem], bufs: dict, send: bool) -> None:
        comm, stream = self._comms[key], int(self._streams[key].cuda_stream)
        with m2n.group() if len(group) > 1 else nullcontext():
            for it in group:
                if send:
                    src, dst = _desc(it, it.src, bufs.get(it.name)), _desc(it, it.dst, None)
                else:
                    src, dst = _desc(it, it.src, None), _desc(it, it.dst, bufs[it.name])
                self._handle.reshard(comm, src, dst, stream=stream)

    async def _wait(self, events, what: str) -> None:
        deadline = time.monotonic() + self.sync_timeout_s
        while not all(e.query() for e in events):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"nccl_m2n {what} did not complete in {self.sync_timeout_s}s; an M2N transfer cannot be "
                    "aborted in-process, restart the trainer and rollout processes of this channel"
                )
            await asyncio.sleep(0.0005)

    # ------------------------------------------------------------------ rollout

    def _build_rollout_epoch(self) -> None:
        info = self._client.request(("epoch", self._membership))
        slices = self._replicas_slices[0]
        r, t, D = self._replica, self._replica_rank, len(slices)
        plans, sizes, mine = {}, {}, {}
        for st, manifests in sorted(info["manifests"].items()):
            key = (st, r)
            plans[key] = build_channel_plan(manifests, slices)
            digest = plan_digest(plans[key])
            if digest != info["digests"][key]:
                raise RuntimeError(
                    f"nccl_m2n: plan digest of channel {key} is {digest}, trainer has {info['digests'][key]}"
                )
            sizes[key] = len(manifests) + D
            mine[key] = len(manifests) + t
        self._reset_epoch()
        self._init_channels(info["uids"], sizes, mine)
        self._buckets = {k: split_buckets(plans[k], self.bucket_size, self.group_bytes) for k in plans}
        if not self._recv_bufs:
            self._recv_bufs = [torch.empty(self.bucket_size, dtype=torch.uint8, device="cuda") for _ in range(2)]
        self._epoch = self._membership

    def _carve(self, buf: torch.Tensor, bucket: list) -> dict[str, torch.Tensor]:
        items = [it for group in bucket for it in group]
        if len(items) == 1 and items[0].dst_bytes > buf.numel():
            it = items[0]
            return {it.name: torch.empty(it.dst.local_shape, dtype=_dtype(it.dtype), device=buf.device)}
        views, off = {}, 0
        for it in items:
            off = (off + _ALIGN - 1) // _ALIGN * _ALIGN
            if off + it.dst_bytes > buf.numel():
                # alignment padding pushed the bucket over: give this item its own tensor
                views[it.name] = torch.empty(it.dst.local_shape, dtype=_dtype(it.dtype), device=buf.device)
                continue
            views[it.name] = buf[off : off + it.dst_bytes].view(_dtype(it.dtype)).view(it.dst.local_shape)
            off += it.dst_bytes
        return views

    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Yield ``(hf_name, tensor)``: this rank's box where the plan shards the tensor, else the
        full tensor. Tensors are views into a reused bucket, valid until the next one is requested."""
        t0 = time.perf_counter()
        if self._epoch != self._membership:
            self._build_rollout_epoch()
        # the trainer must have entered this step's send (with the same membership) before we
        # enqueue the matching reshards
        self._client.request(("go", self._membership, global_steps))

        units = []
        keys = sorted(self._buckets)
        for i in range(max(len(b) for b in self._buckets.values())):
            units += [(k, self._buckets[k][i]) for k in keys if i < len(self._buckets[k])]

        def enqueue(j):
            k, bucket = units[j]
            views = self._carve(self._recv_bufs[j % 2], bucket)
            self._streams[k].wait_stream(torch.cuda.current_stream())
            for group in bucket:
                self._enqueue(k, group, views, send=False)
            return views, self._streams[k].record_event()

        total = 0
        pending = enqueue(0)
        for j in range(len(units)):
            views, event = pending
            pending = enqueue(j + 1) if j + 1 < len(units) else None
            await self._wait([event], "receive")
            for name, tensor in views.items():
                total += tensor.numel() * tensor.element_size()
                yield name, tensor
            # the consumer may copy asynchronously; the buffer is refilled two units later
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        logger.info(
            "nccl_m2n receive: replica %s rank %s, %.2f GB in %.3fs", self._replica, self._replica_rank, total / 1e9, dt
        )
