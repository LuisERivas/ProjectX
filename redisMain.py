# main.py
#
# Redis-only FastAPI gateway for your job system:
# - POST /v1/jobs              -> enqueue job into Redis Stream
# - GET  /v1/jobs/{id}         -> fetch job status/result/error (stored as Redis HASH)
# - GET  /v1/jobs/{id}/events  -> SSE stream of per-job events (tails per-job Redis Stream)
# - GET  /health
#
# Uses Redis OSS (no RedisJSON module required).
#
# Run:
#   pip install -r requirements.txt
#   uvicorn main:app --host 0.0.0.0 --port 8000
#
# Env:
#   REDIS_URL=redis://127.0.0.1:6379/0
#   JOB_TTL_S=3600
#   QUEUE_STREAM_KEY=jobs:stream
#   WORKER_GROUP=workers
#
#   BACKPRESSURE_MAX_BACKLOG=200   (optional; 0 disables)
#     - Backlog = pending (delivered but unacked) + lag (if Redis provides it)
#
#   SSE_BLOCK_MS=15000
#   SSE_HEARTBEAT_S=15

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

try:
    import orjson  # type: ignore
except Exception:
    orjson = None

import redis.asyncio as redis


TaskType = Literal["chat", "plan", "code", "tool", "rag", "embed"]
StatusType = Literal["queued", "running", "done", "error", "canceled"]


def now_ms() -> int:
    return int(time.time() * 1000)


def dumps(obj: Any) -> str:
    if orjson:
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def loads(s: str) -> Any:
    if orjson:
        return orjson.loads(s)
    return json.loads(s)


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

QUEUE_STREAM_KEY = os.getenv("QUEUE_STREAM_KEY", "jobs:stream")
WORKER_GROUP = os.getenv("WORKER_GROUP", "workers")

JOB_TTL_S = env_int("JOB_TTL_S", 3600)

# Option B: backpressure by backlog (pending + lag), NOT XLEN
BACKPRESSURE_MAX_BACKLOG = env_int("BACKPRESSURE_MAX_BACKLOG", 200)  # 0 disables

SSE_BLOCK_MS = env_int("SSE_BLOCK_MS", 15000)
SSE_HEARTBEAT_S = env_int("SSE_HEARTBEAT_S", 15)


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def events_key(job_id: str) -> str:
    return f"job:{job_id}:events"


class CreateJobRequest(BaseModel):
    task: TaskType
    payload: Dict[str, Any] = Field(default_factory=dict)
    ttl_s: Optional[int] = None

    @model_validator(mode="after")
    def validate_payload(self) -> "CreateJobRequest":
        try:
            encoded = dumps(self.payload)
        except Exception as e:
            raise ValueError(f"payload must be JSON-serializable: {e}") from e

        if len(encoded) > 200_000:  # ~200 KB
            raise ValueError("payload too large (max ~200KB for gateway)")
        return self


app = FastAPI(title="Redis FastAPI Gateway", version="0.2.0")

rds: redis.Redis | None = None


async def redis_client() -> redis.Redis:
    global rds
    if rds is None:
        rds = redis.from_url(REDIS_URL, decode_responses=True)
    return rds


async def ensure_worker_group(r: redis.Redis) -> None:
    """
    Ensure the queue stream + consumer group exist.
    Safe to call on startup and on errors.
    """
    try:
        # XGROUP CREATE <key> <group> $ MKSTREAM
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
    data: Dict[str, Any] | None = None,
    ttl_s: int,
) -> str:
    ek = events_key(job_id)
    fields = {
        "type": type_,
        "ts": str(now_ms()),
        "step": step,
        "data": dumps(data or {}),
    }
    event_id = await r.xadd(ek, fields)
    await r.expire(ek, ttl_s)
    return event_id


# ----------------------------
# Job state stored as Redis HASH
# ----------------------------

JOB_HASH_JSON_FIELDS = {"payload", "result", "error"}


def _serialize_job_hash(job_obj: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert python job dict -> Redis hash mapping of strings.
    Complex fields are stored as JSON strings.
    """
    out: Dict[str, str] = {}
    for k, v in job_obj.items():
        if k in JOB_HASH_JSON_FIELDS:
            out[k] = dumps(v)
        else:
            out[k] = "" if v is None else str(v)
    return out


def _deserialize_job_hash(h: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert Redis hash mapping -> python job dict.
    Parses JSON fields back into objects, coerces known numeric fields.
    """
    job: Dict[str, Any] = dict(h)

    # Parse JSON fields
    for k in JOB_HASH_JSON_FIELDS:
        raw = job.get(k)
        if raw is None or raw == "":
            job[k] = None if k in ("result", "error") else {}
            continue
        if isinstance(raw, str):
            try:
                job[k] = loads(raw)
            except Exception:
                job[k] = {"_raw": raw}

    # Coerce numeric fields
    for nk in ("created_ts", "updated_ts", "ttl_s"):
        if nk in job and isinstance(job[nk], str) and job[nk] != "":
            try:
                job[nk] = int(job[nk])
            except Exception:
                pass

    return job


async def job_exists(r: redis.Redis, job_id: str) -> bool:
    # EXISTS works for hash keys too
    return bool(await r.exists(job_key(job_id)))


async def get_job(r: redis.Redis, job_id: str) -> Dict[str, Any]:
    h = await r.hgetall(job_key(job_id))
    if not h:
        raise KeyError("job not found")
    return _deserialize_job_hash(h)


async def set_job(r: redis.Redis, job_id: str, job_obj: Dict[str, Any], ttl_s: int) -> None:
    hk = job_key(job_id)
    mapping = _serialize_job_hash(job_obj)
    await r.hset(hk, mapping=mapping)
    await r.expire(hk, ttl_s)


# ----------------------------
# Backpressure: Option B (backlog), NOT XLEN
# ----------------------------

async def get_queue_backlog(r: redis.Redis) -> Dict[str, Any]:
    """
    Backlog tries to represent "how behind are workers?" without using XLEN.

    We compute:
      pending = XPENDING count (delivered to group but not acked)
      lag = XINFO GROUPS lag (if available in your Redis version)
      backlog = pending + lag (if lag present), else pending

    Note: lag semantics vary by Redis version. If absent, we fall back gracefully.
    """
    await ensure_worker_group(r)

    pending_count: int = 0
    lag: Optional[int] = None

    try:
        # redis-py typically returns dict: {'pending': int, 'min': str, 'max': str, 'consumers': [...]}
        p = await r.xpending(QUEUE_STREAM_KEY, WORKER_GROUP)
        if isinstance(p, dict):
            pending_count = int(p.get("pending", 0) or 0)
        elif isinstance(p, (list, tuple)) and p:
            # Older format: (count, min, max, consumers)
            pending_count = int(p[0] or 0)
    except Exception:
        pending_count = 0

    try:
        groups = await r.xinfo_groups(QUEUE_STREAM_KEY)
        # groups is a list of dicts; find our group
        if isinstance(groups, list):
            for g in groups:
                if g.get("name") == WORKER_GROUP:
                    if "lag" in g and g["lag"] is not None:
                        lag = int(g["lag"])
                    break
    except Exception:
        lag = None

    backlog = pending_count + (lag if lag is not None else 0)

    return {"pending": pending_count, "lag": lag, "backlog": backlog}


async def check_backpressure(r: redis.Redis) -> None:
    if BACKPRESSURE_MAX_BACKLOG <= 0:
        return

    stats = await get_queue_backlog(r)
    backlog = int(stats.get("backlog") or 0)

    if backlog >= BACKPRESSURE_MAX_BACKLOG:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "backpressure",
                "message": "Workers are behind (backlog too high), retry shortly.",
                "details": {
                    "backlog": backlog,
                    "pending": stats.get("pending"),
                    "lag": stats.get("lag"),
                    "max": BACKPRESSURE_MAX_BACKLOG,
                },
            },
            headers={"Retry-After": "1"},
        )


@app.on_event("startup")
async def on_startup() -> None:
    r = await redis_client()
    await r.ping()
    await ensure_worker_group(r)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global rds
    if rds is not None:
        await rds.aclose()
        rds = None


@app.get("/health")
async def health() -> JSONResponse:
    r = await redis_client()
    await r.ping()

    backlog_stats = await get_queue_backlog(r)

    # XLEN is still shown for curiosity/debugging, but NOT used for backpressure.
    try:
        stream_len = await r.xlen(QUEUE_STREAM_KEY)
    except Exception:
        stream_len = None

    return JSONResponse(
        {
            "ok": True,
            "redis_url": REDIS_URL,
            "queue_stream": QUEUE_STREAM_KEY,
            "worker_group": WORKER_GROUP,
            "stream_len": stream_len,
            "backlog": backlog_stats,
            "ts": now_ms(),
        }
    )


@app.post("/v1/jobs")
async def create_job(req: Request, body: CreateJobRequest) -> JSONResponse:
    r = await redis_client()
    await check_backpressure(r)

    ttl_s = body.ttl_s or JOB_TTL_S
    if ttl_s <= 0:
        ttl_s = JOB_TTL_S

    # Optional idempotency
    idem_key = req.headers.get("Idempotency-Key")
    if idem_key:
        idem_redis_key = f"idempotency:{idem_key}"
        existing = await r.get(idem_redis_key)
        if existing:
            return JSONResponse({"job_id": existing, "idempotent": True})

    job_id = str(uuid.uuid4())
    created = now_ms()

    job_obj: Dict[str, Any] = {
        "job_id": job_id,
        "task": body.task,
        "payload": body.payload,
        "status": "queued",
        "created_ts": created,
        "updated_ts": created,
        "ttl_s": ttl_s,
        "result": None,
        "error": None,
    }

    # Store job state as HASH + TTL
    await set_job(r, job_id, job_obj, ttl_s)

    # Seed per-job event stream + TTL
    await emit_event(
        r,
        job_id,
        type_="queued",
        step="gateway.enqueue",
        data={},
        ttl_s=ttl_s,
    )

    # Enqueue into main stream (no MAXLEN here because you asked for Option B)
    await ensure_worker_group(r)
    await r.xadd(
        QUEUE_STREAM_KEY,
        {"job_id": job_id, "task": body.task, "payload": dumps(body.payload)},
    )

    # Set idempotency mapping after enqueue succeeds
    if idem_key:
        await r.set(f"idempotency:{idem_key}", job_id, ex=ttl_s)

    return JSONResponse({"job_id": job_id})


@app.get("/v1/jobs/{job_id}")
async def read_job(job_id: str) -> JSONResponse:
    r = await redis_client()
    try:
        job = await get_job(r, job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "Job not found", "details": {"job_id": job_id}},
        )
    return JSONResponse(job)


def sse_format(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


@app.get("/v1/jobs/{job_id}/events")
async def stream_events(job_id: str, request: Request) -> Response:
    r = await redis_client()

    # Validate job exists (fast fail)
    if not await job_exists(r, job_id):
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "Job not found", "details": {"job_id": job_id}},
        )

    ek = events_key(job_id)

    async def gen() -> AsyncGenerator[bytes, None]:
        last_id = "0-0"
        last_heartbeat = time.time()

        yield sse_format("hello", dumps({"job_id": job_id, "ts": now_ms()}))

        while True:
            if await request.is_disconnected():
                break

            now = time.time()
            if now - last_heartbeat >= SSE_HEARTBEAT_S:
                yield b": heartbeat\n\n"
                last_heartbeat = now

            try:
                resp = await r.xread({ek: last_id}, count=50, block=SSE_BLOCK_MS)
            except redis.ResponseError:
                await asyncio.sleep(0.05)
                continue

            if not resp:
                continue

            _, entries = resp[0]
            for entry_id, fields in entries:
                last_id = entry_id

                event_type = fields.get("type", "message")
                payload = {
                    "id": entry_id,
                    "type": event_type,
                    "ts": int(fields.get("ts") or 0),
                    "step": fields.get("step", ""),
                    "data": None,
                }
                d = fields.get("data")
                if isinstance(d, str) and d:
                    try:
                        payload["data"] = loads(d)
                    except Exception:
                        payload["data"] = {"_raw": d}

                yield sse_format(event_type, dumps(payload))

                if event_type in ("done", "error", "canceled"):
                    return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )