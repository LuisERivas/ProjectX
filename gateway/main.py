from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncGenerator, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from contract.gateway.service import GatewayContract, now_ms
from contract.shared import config as shared_config
from contract.shared.serde import dumps, loads


TaskType = Literal["chat", "plan", "code", "tool", "rag", "embed"]
StatusType = Literal["queued", "running", "done", "error", "canceled"]


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


JOB_TTL_S = env_int("JOB_TTL_S", 3600)
BACKPRESSURE_MAX_BACKLOG = env_int("BACKPRESSURE_MAX_BACKLOG", 200)  # 0 disables
SSE_BLOCK_MS = env_int("SSE_BLOCK_MS", 15000)
SSE_HEARTBEAT_S = env_int("SSE_HEARTBEAT_S", 15)


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


class CancelJobRequest(BaseModel):
    reason: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Redis FastAPI Gateway", version="0.2.0")
gateway_contract = GatewayContract()


async def check_backpressure() -> None:
    if BACKPRESSURE_MAX_BACKLOG <= 0:
        return

    stats = await gateway_contract.get_queue_backlog()
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
    await gateway_contract.ping()
    await gateway_contract.ensure_worker_group()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await gateway_contract.close()


@app.get("/health")
async def health() -> JSONResponse:
    await gateway_contract.ping()
    backlog_stats = await gateway_contract.get_queue_backlog()
    stream_len = await gateway_contract.queue_stream_len()
    return JSONResponse(
        {
            "ok": True,
            "redis_url": shared_config.REDIS_URL,
            "queue_stream": shared_config.QUEUE_STREAM_KEY,
            "worker_group": shared_config.WORKER_GROUP,
            "stream_len": stream_len,
            "backlog": backlog_stats,
            "ts": now_ms(),
        }
    )


@app.post("/v1/jobs")
async def create_job(req: Request, body: CreateJobRequest) -> JSONResponse:
    await check_backpressure()

    ttl_s = body.ttl_s or JOB_TTL_S
    if ttl_s <= 0:
        ttl_s = JOB_TTL_S

    result = await gateway_contract.create_job(
        task=body.task,
        payload=body.payload,
        ttl_s=ttl_s,
        idem_key=req.headers.get("Idempotency-Key"),
    )
    return JSONResponse(result)


@app.get("/v1/jobs/{job_id}")
async def read_job(job_id: str) -> JSONResponse:
    try:
        job = await gateway_contract.read_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "Job not found", "details": {"job_id": job_id}},
        )
    return JSONResponse(job)


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, body: CancelJobRequest) -> JSONResponse:
    try:
        result = await gateway_contract.cancel_job(job_id, reason=body.reason)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "Job not found", "details": {"job_id": job_id}},
        )
    return JSONResponse(result)


def sse_format(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


@app.get("/v1/jobs/{job_id}/events")
async def stream_events(job_id: str, request: Request) -> Response:
    if not await gateway_contract.job_exists(job_id):
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": "Job not found", "details": {"job_id": job_id}},
        )

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
                entries = await gateway_contract.stream_events_batch(
                    job_id,
                    last_id=last_id,
                    count=50,
                    block_ms=SSE_BLOCK_MS,
                )
            except Exception:
                await asyncio.sleep(0.05)
                continue

            if not entries:
                continue

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

