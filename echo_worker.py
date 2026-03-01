#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import traceback
from typing import Any, Dict, Optional, Tuple, List

import redis.asyncio as redis

# --- must match gateway defaults/env ---
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_STREAM_KEY = os.getenv("QUEUE_STREAM_KEY", "jobs:stream")
WORKER_GROUP = os.getenv("WORKER_GROUP", "workers")

# --- worker runtime tuning ---
CONSUMER = os.getenv("CONSUMER", f"echo-{os.getpid()}")
BLOCK_MS = int(os.getenv("BLOCK_MS", "5000"))
COUNT = int(os.getenv("COUNT", "10"))
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "4"))
DEFAULT_TTL_S = int(os.getenv("DEFAULT_TTL_S", "3600"))

# --- reliability hardening ---
# Reclaim messages that have been idle (unacked) for this long.
CLAIM_IDLE_MS = int(os.getenv("CLAIM_IDLE_MS", "60000"))  # 60s
# How many pending messages to attempt to reclaim per pass.
CLAIM_COUNT = int(os.getenv("CLAIM_COUNT", "25"))
# How often to run the pending recovery loop.
CLAIM_EVERY_S = float(os.getenv("CLAIM_EVERY_S", "2.0"))

# Per-job timeout (0 disables). This is wall-clock time for processing_one().
JOB_TIMEOUT_S = float(os.getenv("JOB_TIMEOUT_S", "0"))

# Dead-letter queue
DLQ_STREAM_KEY = os.getenv("DLQ_STREAM_KEY", "jobs:dlq")
DLQ_MAXLEN = int(os.getenv("DLQ_MAXLEN", "10000"))  # approximate cap; best-effort trim

_shutdown = asyncio.Event()


def now_ms() -> int:
    return int(time.time() * 1000)


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def events_key(job_id: str) -> str:
    return f"job:{job_id}:events"


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def loads(s: str) -> Any:
    return json.loads(s)


async def ensure_worker_group(r: redis.Redis) -> None:
    """
    Match gateway contract: id="$" and MKSTREAM.
    """
    try:
        await r.xgroup_create(
            name=QUEUE_STREAM_KEY,
            groupname=WORKER_GROUP,
            id="$",
            mkstream=True,
        )
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            return
        raise


async def emit_event(
    r: redis.Redis,
    job_id: str,
    *,
    type_: str,
    step: str,
    data: Optional[Dict[str, Any]] = None,
    ttl_s: int,
) -> str:
    """
    Event schema: type, ts, step, data (JSON string)
    """
    ek = events_key(job_id)
    fields: Dict[str, str] = {
        "type": type_,
        "ts": str(now_ms()),
        "step": step,
        "data": dumps(data or {}),
    }
    event_id = await r.xadd(ek, fields)
    await r.expire(ek, ttl_s)
    return event_id


async def get_job_hash(r: redis.Redis, job_id: str) -> Dict[str, str]:
    h = await r.hgetall(job_key(job_id))
    return h or {}


async def set_job_fields(r: redis.Redis, job_id: str, fields: Dict[str, Any], ttl_s: int) -> None:
    """
    Must match gateway hash serialization:
      JSON fields: payload, result, error => JSON strings
      everything else => str
    """
    hk = job_key(job_id)
    mapping: Dict[str, str] = {}

    for k, v in fields.items():
        if k in ("payload", "result", "error"):
            mapping[k] = "" if v is None else dumps(v)
        else:
            mapping[k] = "" if v is None else str(v)

    await r.hset(hk, mapping=mapping)
    await r.expire(hk, ttl_s)


async def push_dlq(
    r: redis.Redis,
    *,
    msg_id: str,
    msg_fields: Dict[str, str],
    error: str,
    tb: str,
) -> None:
    """
    Store poison messages for inspection/replay.
    """
    fields: Dict[str, str] = {
        "src_stream": QUEUE_STREAM_KEY,
        "src_group": WORKER_GROUP,
        "src_msg_id": msg_id,
        "consumer": CONSUMER,
        "ts": str(now_ms()),
        "error": error,
        "traceback": tb,
        "job_id": (msg_fields.get("job_id") or "").strip(),
        "task": (msg_fields.get("task") or "").strip(),
        "payload": msg_fields.get("payload") or "",
        "raw": dumps(msg_fields),
    }
    # best-effort approximate trim
    try:
        await r.xadd(DLQ_STREAM_KEY, fields, maxlen=DLQ_MAXLEN, approximate=True)
    except TypeError:
        # older redis-py fallback (no approximate kw)
        await r.xadd(DLQ_STREAM_KEY, fields)


def _parse_payload(raw_payload: str) -> Dict[str, Any]:
    try:
        obj = loads(raw_payload or "{}")
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        return {"_raw": raw_payload}


async def process_one(r: redis.Redis, msg_id: str, msg_fields: Dict[str, str]) -> None:
    """
    Queue message fields: job_id, task, payload (payload is JSON string).
    Updates job hash + job events. ACK last.
    """
    job_id = (msg_fields.get("job_id") or "").strip()
    if not job_id:
        # malformed message; ack to avoid wedging backlog
        await r.xack(QUEUE_STREAM_KEY, WORKER_GROUP, msg_id)
        return

    # If job hash doesn't exist, ACK and move on (gateway is source of truth).
    job_h = await get_job_hash(r, job_id)
    if not job_h:
        await r.xack(QUEUE_STREAM_KEY, WORKER_GROUP, msg_id)
        return

    # TTL from job hash if present
    ttl_s = DEFAULT_TTL_S
    try:
        if job_h.get("ttl_s"):
            ttl_s = int(job_h["ttl_s"])
    except Exception:
        ttl_s = DEFAULT_TTL_S

    task = (msg_fields.get("task") or "").strip()
    payload_obj = _parse_payload(msg_fields.get("payload") or "{}")

    started = now_ms()

    # Transition to running
    await set_job_fields(
        r,
        job_id,
        {"status": "running", "updated_ts": started, "error": None, "result": None},
        ttl_s=ttl_s,
    )
    await emit_event(
        r,
        job_id,
        type_="running",
        step="worker.start",
        data={"consumer": CONSUMER, "task": task},
        ttl_s=ttl_s,
    )

    try:
        # --- echo logic (placeholder; replace later with local model call) ---
        result_text = f"echo(task={task}): {payload_obj}"

        await emit_event(
            r,
            job_id,
            type_="message",
            step="worker.echo",
            data={"text": result_text},
            ttl_s=ttl_s,
        )

        done_ts = now_ms()

        # Transition to done
        await set_job_fields(
            r,
            job_id,
            {
                "status": "done",
                "updated_ts": done_ts,
                "result": {"text": result_text, "ms": done_ts - started},
                "error": None,
            },
            ttl_s=ttl_s,
        )

        # Terminal event for SSE termination
        await emit_event(
            r,
            job_id,
            type_="done",
            step="worker.done",
            data={"ms": done_ts - started},
            ttl_s=ttl_s,
        )

        # ACK only after state + terminal event are committed
        await r.xack(QUEUE_STREAM_KEY, WORKER_GROUP, msg_id)

    except asyncio.CancelledError:
        # Best-effort: do not ACK; allow claim/retry by another consumer.
        raise
    except Exception as e:
        err_ts = now_ms()
        err_msg = str(e)
        tb = traceback.format_exc(limit=20)

        await set_job_fields(
            r,
            job_id,
            {"status": "error", "updated_ts": err_ts, "error": {"message": err_msg}},
            ttl_s=ttl_s,
        )
        await emit_event(
            r,
            job_id,
            type_="error",
            step="worker.error",
            data={"message": err_msg},
            ttl_s=ttl_s,
        )

        # DLQ + ACK to avoid wedging backlog; job is terminal error anyway.
        await push_dlq(r, msg_id=msg_id, msg_fields=msg_fields, error=err_msg, tb=tb)
        await r.xack(QUEUE_STREAM_KEY, WORKER_GROUP, msg_id)


async def _process_one_with_timeout(
    r: redis.Redis, msg_id: str, msg_fields: Dict[str, str]
) -> None:
    if JOB_TIMEOUT_S and JOB_TIMEOUT_S > 0:
        await asyncio.wait_for(process_one(r, msg_id, msg_fields), timeout=JOB_TIMEOUT_S)
    else:
        await process_one(r, msg_id, msg_fields)


async def pending_recovery_loop(r: redis.Redis, sem: asyncio.Semaphore, inflight: "set[asyncio.Task]") -> None:
    """
    Reclaim stuck pending messages (e.g., worker crash after delivery, before ACK).
    Uses XAUTOCLAIM to transfer ownership to this consumer, then processes them.
    """
    start_id = "0-0"
    while not _shutdown.is_set():
        try:
            # Don't reclaim if we have no capacity.
            if sem.locked():
                await asyncio.sleep(CLAIM_EVERY_S)
                continue

            res = await r.xautoclaim(
                name=QUEUE_STREAM_KEY,
                groupname=WORKER_GROUP,
                consumername=CONSUMER,
                min_idle_time=CLAIM_IDLE_MS,
                start_id=start_id,
                count=CLAIM_COUNT,
            )
            # redis-py returns: (next_start_id, messages, deleted_ids)
            next_start_id, messages, _deleted = res
            start_id = next_start_id or "0-0"

            if not messages:
                await asyncio.sleep(CLAIM_EVERY_S)
                continue

            for mid, mfields in messages:
                # Gate on semaphore; ensures we don't exceed MAX_INFLIGHT
                await sem.acquire()

                async def _run(mid_: str, mf_: Dict[str, str]) -> None:
                    try:
                        await _process_one_with_timeout(r, mid_, mf_)
                    finally:
                        sem.release()

                t = asyncio.create_task(_run(mid, mfields))
                inflight.add(t)
                t.add_done_callback(lambda tt: inflight.discard(tt))

        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(CLAIM_EVERY_S)


async def run() -> None:
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await r.ping()
    await ensure_worker_group(r)

    sem = asyncio.Semaphore(MAX_INFLIGHT)
    inflight: set[asyncio.Task] = set()

    async def spawn(mid: str, mfields: Dict[str, str]) -> None:
        await sem.acquire()

        async def _run() -> None:
            try:
                await _process_one_with_timeout(r, mid, mfields)
            finally:
                sem.release()

        t = asyncio.create_task(_run())
        inflight.add(t)
        t.add_done_callback(lambda tt: inflight.discard(tt))

    # Start pending recovery loop
    recovery_task = asyncio.create_task(pending_recovery_loop(r, sem, inflight))

    try:
        while not _shutdown.is_set():
            try:
                # If we are saturated, wait until capacity returns before reading more.
                if sem.locked():
                    await asyncio.sleep(0.01)
                    continue

                resp = await r.xreadgroup(
                    groupname=WORKER_GROUP,
                    consumername=CONSUMER,
                    streams={QUEUE_STREAM_KEY: ">"},
                    count=COUNT,
                    block=BLOCK_MS,
                )
                if not resp:
                    continue

                for _stream, entries in resp:
                    for mid, mfields in entries:
                        await spawn(mid, mfields)

            except asyncio.CancelledError:
                break
            except Exception:
                # brief backoff on transient errors
                await asyncio.sleep(0.25)

    finally:
        _shutdown.set()
        recovery_task.cancel()
        with contextlib.suppress(Exception):
            await recovery_task

        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        await r.aclose()


def _signal_handler(*_args: object) -> None:
    _shutdown.set()


if __name__ == "__main__":
    import contextlib

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    asyncio.run(run())