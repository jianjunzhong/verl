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

Channels: one communicator + stream per (trainer PP stage, rollout replica). A replica is
identified by the actor ids of its workers, so it keeps its channels for as long as its processes
live, whatever else joins or leaves.

Receivers: by default the checkpoint engine worker receives and hands named tensors to its rollout
server. A rollout whose layout lists the ``nccl_m2n`` wire format receives in its own workers
instead (vLLM: ``update_weights_from_m2n``): the checkpoint engine worker only forwards the
channel info, and the workers write most tensors in place (:class:`M2NReceiver` ``direct``).

Lifecycle:

* ``prepare`` / ``build_topology`` / ``init_process_group`` run on every update but only record
  the replica set of that update (and a fresh update token). Nothing is torn down.
* On the first send, trainer ranks all-gather their manifests (once per process).
* Every send diffs the replica set against the known channels: a new replica gets its channels
  (unique ids from trainer rank 0, plans built and digest-checked on both sides); a replica that
  left is parked, i.e. skipped with its communicator kept alive; a parked replica that comes back
  reuses its communicator. The other replicas are not touched.
* Trainer rank 0 publishes the per-replica channel info of the update over a ZMQ REP socket;
  rollout ranks ask for it with the update token, which also orders them behind the trainer.
* Steady state: one ``Handle.reshard`` per tensor straight from the source view, stream-ordered,
  no staging window, no per-weight host sync and no per-weight metadata. Receivers fill
  double-buffered buckets and yield them; senders keep each channel at most two buckets ahead
  and wait for one event per channel at the end.

nccl-extensions 0.1.0 rules this follows:

1. A communicator must not be destroyed while an M2N handle lives (the next communicator allocated
   at that address breaks), so a replica that leaves is parked instead of closed.
2. A handle reset must be symmetric. When the communicators of one handle epoch would exceed
   ``max_channels``, the trainer starts a new generation: every rank destroys its handle, aborts
   its communicators and rebuilds the channels of the current replicas. Parked replicas notice
   the new generation when they come back.
3. Once a transfer has started, a dead peer hangs every survivor of its communicator and M2N
   cannot abort it in-process. A sync that misses ``sync_timeout_s`` raises and fences the engine
   (every later transfer raises); the processes of that channel, including the trainer sender,
   must be restarted. Channels to other replicas are separate communicators and complete.
4. On NVSwitch nodes, M2N windows need NVLS multicast groups, which vLLM's own communicators also
   take; with receivers inside vLLM workers the pool runs out (``ncclCommWindowRegister ... out of
   memory``). Set ``NCCL_NVLS_ENABLE=0`` for the trainer and rollout processes there.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Iterator, Optional

import ray
import torch
import torch.distributed as dist
import zmq

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_m2n_plan import (
    Piece,
    PlanItem,
    Side,
    build_channel_plan,
    plan_digest,
    split_buckets,
    unstack,
)
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

try:
    import nccl.m2n as m2n
    from nccl.core import Communicator, UniqueId, get_unique_id
    from nccl.m2n import DistTensor, Handle, Mesh, Replicate, Shard
except ImportError as e:  # pragma: no cover - depends on nccl-extensions
    raise ImportError("checkpoint_engine.backend=nccl_m2n needs nccl-extensions (pip install nccl-extensions)") from e

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["NCCLM2NCheckpointEngine", "M2NReceiver", "WIRE_FORMAT"]

# rollout workers that receive M2N transfers themselves advertise this in RolloutLayout.wire_formats
WIRE_FORMAT = "nccl_m2n"
# every bucket view starts on this byte alignment
_ALIGN = 256


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _digest(obj) -> str:
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()[:16]


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class _Server:
    """Trainer rank 0's control plane: answers ``("plan", update, replica, manifest_digest)`` with
    that replica's channel info once the trainer has started sending ``update``."""

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
        self._plan: Optional[tuple[str, dict]] = None
        self._thread = threading.Thread(target=self._serve, name="nccl_m2n_ctl", daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> dict:
        return {"ip": self.ip, "port": self.port}

    def publish(self, update: str, info: dict) -> None:
        with self._cv:
            self._plan = (update, info)
            self._cv.notify_all()

    def _serve(self) -> None:
        while True:
            try:
                _kind, update, replica, manifest_digest = self._sock.recv_pyobj()
            except zmq.ZMQError:
                return
            with self._cv:

                def ready(u=update):
                    return self._plan is not None and self._plan[0] == u

                rep = None
                if self._cv.wait_for(ready, timeout=self.timeout_s):
                    info = self._plan[1]
                    channel = info["channels"].get(replica)
                    if channel is None:
                        rep = {"error": f"replica {replica} is not part of update {update}"}
                    else:
                        rep = {"gen": info["gen"], "channel": channel, "manifest_digest": info["manifest_digest"]}
                        if manifest_digest != info["manifest_digest"]:
                            rep["manifests"] = info["manifests"]
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
        if "error" in rep:
            raise RuntimeError(f"nccl_m2n: {rep['error']}")
        return rep


@dataclass
class _Channel:
    comm: Any
    stream: torch.cuda.Stream
    buckets: list


def _desc(item: PlanItem, side: Side, buf: Optional[torch.Tensor]) -> DistTensor:
    if side.shard_dim is None:
        mesh, placements = Mesh((side.n, 1), start_rank=side.start), [Replicate(), Replicate()]
    else:
        # row-major: rank start + i * shard_n + j holds shard j of the i-th replica of the side
        mesh = Mesh((side.repl, side.shard_n), start_rank=side.start)
        placements = [Replicate(), Shard(side.shard_dim)]
    return DistTensor(buf, local_shape=side.local_shape, dtype=_dtype(item.dtype), mesh=mesh, placements=placements)


def _timeout(what: str, timeout_s: float) -> TimeoutError:
    return TimeoutError(
        f"nccl_m2n {what} did not complete in {timeout_s}s; an M2N transfer cannot be aborted in-process, "
        "restart the trainer and rollout processes of this channel"
    )


async def _await_events(events, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not all(e.query() for e in events):
        if time.monotonic() > deadline:
            raise _timeout(what, timeout_s)
        await asyncio.sleep(0.0005)


def _block_events(events, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not all(e.query() for e in events):
        if time.monotonic() > deadline:
            raise _timeout(what, timeout_s)
        time.sleep(0.0005)


class _Endpoint:
    """This process's M2N handle and channel communicators of one handle epoch."""

    def __init__(self, bucket_size: int, group_bytes: int):
        self.bucket_size = bucket_size
        self.group_bytes = group_bytes
        self.handle = None
        self.channels: dict[Any, _Channel] = {}
        self.opened = 0

    def open(self, key, uid: bytes, size: int, rank: int, plan: list[PlanItem]) -> None:
        if self.handle is None:
            self.handle = Handle.create()
        comm = Communicator.init(size, rank, UniqueId.from_bytes(uid))
        buckets = split_buckets(plan, self.bucket_size, self.group_bytes)
        self.channels[key] = _Channel(comm, torch.cuda.Stream(), buckets)
        self.opened += 1

    def close(self) -> None:
        """Destroy the handle, then abort (local, no peer needed) every communicator of it."""
        if self.handle is None and not self.channels:
            return
        torch.cuda.synchronize()
        if self.handle is not None:
            self.handle.destroy()
            self.handle = None
        for ch in self.channels.values():
            ch.comm.abort()
        self.channels = {}

    def enqueue(self, ch: _Channel, group: list[PlanItem], bufs: dict, send: bool) -> None:
        stream = int(ch.stream.cuda_stream)
        with m2n.group() if len(group) > 1 else nullcontext():
            for it in group:
                if send:
                    src, dst = _desc(it, it.src, bufs.get(it.name)), _desc(it, it.dst, None)
                else:
                    src, dst = _desc(it, it.src, None), _desc(it, it.dst, bufs[it.name])
                self.handle.reshard(ch.comm, src, dst, stream=stream)


# returns the tensor an item may be received into in place (contiguous, its local shape and dtype), or None
DirectFn = Callable[[PlanItem], Optional[torch.Tensor]]


class M2NReceiver:
    """One rollout rank's end of its replica's channels (one per trainer stage).

    ``sync`` applies the trainer's channel info of an update (opening the channels on a new
    replica or generation); ``receive`` / ``areceive`` run the transfer in double-buffered units
    and yield each unit's staged ``(name, tensor)`` list (views valid until the next unit). With
    ``direct``, items that have an in-place destination are written there and not yielded.

    Args:
        keep_buffers: keep the two staging buffers between updates (else they are freed after each).
    """

    def __init__(self, bucket_size: int, group_bytes: int, timeout_s: float, keep_buffers: bool = True):
        self.ep = _Endpoint(bucket_size, group_bytes)
        self.timeout_s = timeout_s
        self.keep_buffers = keep_buffers
        self.gen: Optional[int] = None
        self.uids: Optional[dict] = None
        self.manifests: Optional[dict] = None
        self.manifest_digest: Optional[str] = None
        self.broken: Optional[str] = None
        self.last_counts = {"direct": 0, "staged": 0, "bytes": 0}
        self.rank: Optional[int] = None
        self._bufs: list[torch.Tensor] = []
        self._items: dict[str, PlanItem] = {}

    def take_manifests(self, rep: dict) -> None:
        if "manifests" in rep:
            self.manifests, self.manifest_digest = rep["manifests"], rep["manifest_digest"]

    def sync(self, rep: dict, slices: list, replica_rank: int) -> None:
        """Apply a ``("plan", ...)`` reply; ``slices``: every rank's ``RolloutLayout.slices`` in TP order."""
        if self.broken is not None:
            raise RuntimeError(f"nccl_m2n receiver is unusable after a failed transfer ({self.broken}); restart it")
        self.take_manifests(rep)
        if self.manifests is None or self.manifest_digest != rep["manifest_digest"]:
            raise RuntimeError("nccl_m2n: this receiver does not have the trainer manifests of the update")
        info = rep["channel"]
        self.rank = replica_rank
        if rep["gen"] == self.gen and info["uids"] == self.uids:
            return
        # new replica, or the trainer started a new generation (possibly while this one was parked)
        self.ep.close()
        self.gen, self.uids = rep["gen"], None
        for st, manifests in sorted(self.manifests.items()):
            plan = build_channel_plan(manifests, slices)
            digest, expected = plan_digest(plan), info["digests"][st]
            if digest != expected:
                raise RuntimeError(f"nccl_m2n: plan digest of stage {st} is {digest}, trainer has {expected}")
            self.ep.open(st, info["uids"][st], info["sizes"][st], len(manifests) + replica_rank, plan)
        self.uids = info["uids"]
        logger.info(
            "nccl_m2n gen %d: rollout rank %d opened %d channels", self.gen, replica_rank, len(self.ep.channels)
        )

    def _prepare(self, direct: Optional[DirectFn]) -> tuple[list, dict]:
        units, chans = [], self.ep.channels
        keys = sorted(chans)
        for i in range(max(len(chans[k].buckets) for k in keys)):
            units += [(k, chans[k].buckets[i]) for k in keys if i < len(chans[k].buckets)]
        targets, need = {}, 0
        self._items = {}
        for _k, bucket in units:
            staged = 0
            for group in bucket:
                for it in group:
                    self._items[it.name] = it
                    t = direct(it) if direct is not None and it.dst.shard_dim is not None else None
                    if t is not None:
                        targets[it.name] = t
                    else:
                        staged += _align(it.dst_bytes)
            need = max(need, min(staged, self.ep.bucket_size))
        if not self._bufs or self._bufs[0].numel() < need:
            self._bufs = [torch.empty(need, dtype=torch.uint8, device="cuda") for _ in range(2)]
        self.last_counts = {"direct": len(targets), "staged": 0, "bytes": 0}
        return units, targets

    def _enqueue(self, units: list, j: int, targets: dict) -> tuple[dict, Any]:
        k, bucket = units[j]
        ch = self.ep.channels[k]
        buf, staged, off = self._bufs[j % 2], {}, 0
        for group in bucket:
            for it in group:
                if it.name in targets:
                    continue
                off = _align(off)
                if off + it.dst_bytes > buf.numel():
                    # larger than the buffer or pushed over it by alignment: give it its own tensor
                    staged[it.name] = torch.empty(it.dst.local_shape, dtype=_dtype(it.dtype), device=buf.device)
                    continue
                staged[it.name] = buf[off : off + it.dst_bytes].view(_dtype(it.dtype)).view(it.dst.local_shape)
                off += it.dst_bytes
        ch.stream.wait_stream(torch.cuda.current_stream())
        bufs = {**targets, **staged}
        for group in bucket:
            self.ep.enqueue(ch, group, bufs, send=False)
        return staged, ch.stream.record_event()

    def _account(self, staged: dict) -> list:
        self.last_counts["staged"] += len(staged)
        self.last_counts["bytes"] += sum(t.numel() * t.element_size() for t in staged.values())
        return [p for name, t in staged.items() for p in unstack(self._items[name], t, self.rank)]

    def _done(self) -> None:
        torch.cuda.synchronize()
        if not self.keep_buffers:
            self._bufs = []

    def receive(self, direct: Optional[DirectFn] = None) -> Iterator[list[tuple[str, torch.Tensor]]]:
        units, targets = self._prepare(direct)
        try:
            pending = self._enqueue(units, 0, targets) if units else None
            for j in range(len(units)):
                staged, event = pending
                pending = self._enqueue(units, j + 1, targets) if j + 1 < len(units) else None
                _block_events([event], self.timeout_s, "receive")
                yield self._account(staged)
                # the consumer may copy asynchronously; the buffer is refilled two units later
                torch.cuda.synchronize()
        except TimeoutError as e:
            self.broken = f"receive: {e!r}"
            raise
        self._done()

    async def areceive(self, direct: Optional[DirectFn] = None) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        units, targets = self._prepare(direct)
        try:
            pending = self._enqueue(units, 0, targets) if units else None
            for j in range(len(units)):
                staged, event = pending
                pending = self._enqueue(units, j + 1, targets) if j + 1 < len(units) else None
                await _await_events([event], self.timeout_s, "receive")
                yield self._account(staged)
                torch.cuda.synchronize()
        except TimeoutError as e:
            self.broken = f"receive: {e!r}"
            raise
        self._done()


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
        max_channels: communicators a trainer sender keeps per handle epoch, parked ones included
            (nccl-extensions 0.1.0 allows 128 per handle, and fewer with NVLS). A join beyond it
            resets every rank to a new generation.
        stack_experts: ask the trainer engine for stacked MoE expert tensors (one reshard per layer
            and projection kind instead of one per expert) where its layout allows it.
    """

    send_weights_takes_engine = True
    needs_rollout_layout = True

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        group_bytes: int = 0,
        sync_timeout_s: float = 600.0,
        max_channels: int = 32,
        stack_experts: bool = True,
    ) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.group_bytes = int(group_bytes)
        self.sync_timeout_s = float(sync_timeout_s)
        self.max_channels = int(max_channels)
        self.stack_experts = bool(stack_experts)

        self._is_rollout = False
        self._layout = None
        self._direct = False
        self._server = _Server(self.sync_timeout_s) if is_master else None
        self._client: Optional[_Client] = None
        # set when a transfer missed its deadline: its streams may never complete
        self._broken: Optional[str] = None

        # registered by init_process_group for the current update
        self._update: Optional[str] = None
        self._replicas: list[tuple[str, list]] = []  # trainer: (replica id, per-rank slices)
        self._replica: Optional[str] = None  # rollout: own replica id
        self._replica_rank: Optional[int] = None
        self._slices: list = []

        # trainer: manifests are gathered once; channel state lives for one generation
        self._stage: Optional[int] = None
        self._sender: Optional[int] = None
        self._manifests: Optional[dict[int, list[list[Piece]]]] = None
        self._manifest_digest: Optional[str] = None
        self._local_digest: Optional[str] = None
        self._gen = 0
        self._known: dict[str, dict] = {}  # replica id -> {"idx", "uids", "digests", "sizes", "plan"}
        self._active: list[str] = []
        self._ep = _Endpoint(bucket_size, self.group_bytes)

        # rollout
        self._rx: Optional[M2NReceiver] = None
        self._forwarded_digest: Optional[str] = None
        self._slices_sent = False

    @property
    def wire_format(self) -> str:
        return WIRE_FORMAT if self._direct else "named_tensors"

    # ------------------------------------------------------------------ topology

    def set_rollout_layout(self, layout) -> None:
        self._is_rollout = True
        self._layout = layout
        self._direct = layout is not None and WIRE_FORMAT in (layout.wire_formats or [])
        self._rx = M2NReceiver(self.bucket_size, self.group_bytes, self.sync_timeout_s)

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

        # a replica is its DP x TP ranks, in the DP-major, TP-minor order of vLLM's EP group
        groups: list[list[int]] = []
        for i, meta in enumerate(rollout):
            layout = meta["layout"]
            if layout is None:
                raise NotImplementedError(
                    "nccl_m2n needs every rollout worker to report its RolloutLayout (get_weight_ownership)"
                )
            if layout["pp_size"] != 1:
                raise NotImplementedError("nccl_m2n does not support rollout pipeline parallelism yet")
            place = layout.get("dp_rank", 0) * layout["tp_size"] + layout["tp_rank"]
            if place == 0:
                groups.append([])
            if not groups or len(groups[-1]) != place:
                raise ValueError(f"nccl_m2n: rollout worker {i} is out of replica order ({place})")
            groups[-1].append(i)
        for g in groups:
            layout = rollout[g[0]]["layout"]
            if len(g) != layout["tp_size"] * layout.get("dp_size", 1):
                raise ValueError(f"nccl_m2n: incomplete rollout replica {g}")

        update = uuid.uuid4().hex
        ids = [_digest([rollout[i]["actor_id"] for i in g]) for g in groups]
        slices = [[rollout[i]["layout"].get("slices") for i in g] for g in groups]
        replicas = list(zip(ids, slices, strict=True))
        actor_kwargs = {
            "role": ["trainer"] * actor_wg_world_size,
            "master": [master] * actor_wg_world_size,
            "update": [update] * actor_wg_world_size,
            "replicas": [replicas] * actor_wg_world_size,
        }
        rollout_kwargs = {k: [None] * rollout_world_size for k in ("replica", "replica_rank", "slices")}
        for g, rid, sl in zip(groups, ids, slices, strict=True):
            for t, i in enumerate(g):
                rollout_kwargs["replica"][i] = rid
                rollout_kwargs["replica_rank"][i] = t
                rollout_kwargs["slices"][i] = sl
        rollout_kwargs.update(
            role=["rollout"] * rollout_world_size,
            master=[master] * rollout_world_size,
            update=[update] * rollout_world_size,
        )
        return actor_kwargs, rollout_kwargs

    def init_process_group(
        self,
        role: str,
        master: dict,
        update: str,
        replicas: Optional[list] = None,
        replica: Optional[str] = None,
        replica_rank: Optional[int] = None,
        slices: Optional[list] = None,
    ):
        """Record this update's replica set; channels are opened, parked or reused by the transfer."""
        if role == "rollout" and self._client is None:
            self._client = _Client(master, self.sync_timeout_s)
        self._update = update
        if role == "trainer":
            self._replicas = replicas
        else:
            # the layouts of a replica are baked once per process, so its slices only change
            # when the replica itself does
            self._slices_sent = self._slices_sent and replica == self._replica
            self._replica, self._replica_rank, self._slices = replica, replica_rank, slices

    def finalize(self):
        """Keep communicators and buffers: they live as long as their replicas."""

    def channel_state(self) -> dict:
        """Debug/test view: generation, open channels, communicators opened so far."""
        ep = self._rx.ep if self._rx is not None else self._ep
        gen = self._rx.gen if self._rx is not None else self._gen
        return {"gen": gen, "channels": sorted(map(str, ep.channels)), "opened": ep.opened}

    # ------------------------------------------------------------------ trainer

    def _pieces(self, shards) -> list[Piece]:
        return [
            Piece(n, tuple(m.global_shape), _dtype_name(v.dtype), tuple(m.offset), tuple(m.local_shape))
            for n, v, m in shards
        ]

    def _gather_manifests(self, source, shards) -> None:
        stage, sender = getattr(source, "stage", 0), source.sender_index
        pieces = self._pieces(shards) if sender is not None else []
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
        self._stage, self._sender = stage, sender
        self._manifests, self._manifest_digest = manifests, _digest(manifests)
        self._local_digest = _digest(pieces)

    def _check_manifest(self, shards) -> None:
        if self._sender is not None and _digest(self._pieces(shards)) != self._local_digest:
            raise RuntimeError("nccl_m2n: the trainer's local shard layout changed; restart the engine")

    def _sync_membership(self) -> dict:
        """Diff this update's replicas against the known channels; identical on every trainer rank."""
        active = [rid for rid, _ in self._replicas]
        slices = dict(self._replicas)
        prev = set(self._active)
        new = [rid for rid in active if rid not in self._known]
        stats = {
            "left": len(prev - set(active)),
            "rejoined": sum(1 for rid in active if rid in self._known and rid not in prev),
            "reset": 0,
        }
        if len(active) > self.max_channels:
            raise ValueError(f"nccl_m2n: {len(active)} rollout replicas exceed max_channels={self.max_channels}")
        if new and len(self._known) + len(new) > self.max_channels:
            self._gen += 1
            self._known = {}
            new = list(active)
            stats["reset"] = 1
        # replicas whose channels are opened by this update (all of them after a reset)
        stats["opened"] = len(new)

        box = [None]
        if new and self.is_master:
            box = [{rid: {st: bytes(get_unique_id()) for st in self._manifests} for rid in new}]
        if new:
            dist.broadcast_object_list(box, src=0)
        for rid in new:
            D = len(slices[rid])
            plans = {st: build_channel_plan(self._manifests[st], slices[rid]) for st in self._manifests}
            self._known[rid] = {
                "idx": len(self._known),
                "uids": box[0][rid],
                "digests": {st: plan_digest(p) for st, p in plans.items()},
                "sizes": {st: len(self._manifests[st]) + D for st in self._manifests},
                "plan": plans.get(self._stage) if self._sender is not None else None,
                "items": sum(len(p) for p in plans.values()),
            }

        if self._server is not None:
            channels = {rid: {k: self._known[rid][k] for k in ("idx", "uids", "digests", "sizes")} for rid in active}
            self._server.publish(
                self._update,
                {
                    "gen": self._gen,
                    "manifest_digest": self._manifest_digest,
                    "manifests": self._manifests,
                    "channels": channels,
                },
            )
        if stats["reset"]:
            self._ep.close()
        if self._sender is not None:
            # global (replica index) order on every rank, so overlapping inits cannot deadlock
            for rid in sorted(new, key=lambda r: self._known[r]["idx"]):
                k = self._known[rid]
                self._ep.open(rid, k["uids"][self._stage], k["sizes"][self._stage], self._sender, k["plan"])
        self._active = active
        if new or stats["left"] or stats["rejoined"]:
            logger.info(
                "nccl_m2n gen %d: stage %s sender %s, %d active replicas, %d channels open, %s",
                self._gen,
                self._stage,
                self._sender,
                len(active),
                len(self._ep.channels),
                stats,
            )
        return stats

    @torch.no_grad()
    async def send_weights(self, engine, global_steps: int | None = None):
        """Send this rank's local HF shards of ``engine`` to every rollout replica of the update."""
        if self._broken is not None:
            raise RuntimeError(f"nccl_m2n engine is unusable after a failed transfer ({self._broken}); restart it")
        t0 = time.perf_counter()
        source = engine.get_hf_shard_source(stack_experts=self.stack_experts)
        shards = list(source.iter_shards())
        t_export = time.perf_counter()
        if self._manifests is None:
            self._gather_manifests(source, shards)
        else:
            self._check_manifest(shards)
        stats = self._sync_membership()
        t_plan = time.perf_counter()
        metrics = {
            "nccl_m2n/export_s": t_export - t0,
            "nccl_m2n/membership_s": t_plan - t_export,
            **{f"nccl_m2n/{k}": v for k, v in stats.items()},
        }
        if self._active:
            # reshards per sync of one replica, over all stages
            metrics["nccl_m2n/items"] = self._known[self._active[0]]["items"]
        channels = [self._ep.channels[rid] for rid in self._active if rid in self._ep.channels]
        if not channels:
            return metrics

        local = {n: v for n, v, _m in shards}
        cur = torch.cuda.current_stream()
        for ch in channels:
            ch.stream.wait_stream(cur)
        try:
            t_enq = await self._send_windowed(channels, local)
            await _await_events([ch.stream.record_event() for ch in channels], self.sync_timeout_s, "send")
        except BaseException as e:
            self._broken = f"send: {e!r}"
            raise
        metrics.update({"nccl_m2n/enqueue_s": t_enq - t_plan, "nccl_m2n/send_s": time.perf_counter() - t_plan})
        return metrics

    async def _send_windowed(self, channels: list[_Channel], local: dict) -> float:
        """Enqueue every channel's buckets in order, at most two in flight per channel (what its
        receiver posts at a time). Native M2N calls hold one process-wide lock and block while the
        peer does not drain them, so a dead receiver must stall only its own channel, at its window,
        not the other channels or the deadline. Returns when the last bucket is enqueued."""
        deadline = time.monotonic() + self.sync_timeout_s
        sent = {id(ch): [] for ch in channels}
        while True:
            progressed = waiting = False
            for ch in channels:
                events = sent[id(ch)]
                i = len(events)
                if i == len(ch.buckets):
                    continue
                if i >= 2 and not events[i - 2].query():
                    waiting = True
                    continue
                for group in ch.buckets[i]:
                    self._ep.enqueue(ch, group, local, send=True)
                events.append(ch.stream.record_event())
                progressed = True
            if not progressed and not waiting:
                return time.perf_counter()
            if not progressed:
                if time.monotonic() > deadline:
                    raise _timeout("send", self.sync_timeout_s)
                await asyncio.sleep(0.0002)

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, Any], None]:
        """Yield ``(hf_name, tensor)``: this rank's box where the plan shards the tensor, else the
        full tensor; views into a reused bucket, valid until the next one is requested.

        With the ``nccl_m2n`` wire format, yield one ``("nccl_m2n", update_info)`` instead: the
        rollout's own workers receive (``M2NReceiver.sync`` with the unpickled ``update_info["reply"]``)."""
        t0 = time.perf_counter()
        # also waits until the trainer has entered this update's send
        rep = self._client.request(("plan", self._update, self._replica, self._rx.manifest_digest))
        if self._direct:
            self._rx.take_manifests(rep)
            forward = dict(rep)
            if self._forwarded_digest != rep["manifest_digest"]:
                forward["manifests"] = self._rx.manifests
            info = {
                # pickled: vLLM's collective_rpc serializes dataclasses (the manifests) as dicts
                "reply": pickle.dumps(forward),
                "bucket_size": self.bucket_size,
                "group_bytes": self.group_bytes,
                "timeout_s": self.sync_timeout_s,
            }
            if not self._slices_sent:
                # every rank's boxes, in replica order: the workers build the plan from them and
                # cannot gather them themselves (a DP replica spans several engine cores)
                info["slices"] = self._slices
            yield (WIRE_FORMAT, info)
            self._forwarded_digest = rep["manifest_digest"]
            self._slices_sent = True
            return

        self._rx.sync(rep, self._slices, self._replica_rank)
        async for staged in self._rx.areceive():
            for name, tensor in staged:
                yield name, tensor
        logger.info(
            "nccl_m2n receive: replica %s rank %s, %.2f GB in %.3fs",
            self._replica,
            self._replica_rank,
            self._rx.last_counts["bytes"] / 1e9,
            time.perf_counter() - t0,
        )
