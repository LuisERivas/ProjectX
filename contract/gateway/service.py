from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as redis

from contract.shared import config as shared_config
from contract.shared.group_ops import ensure_worker_group
from contract.shared.job_hash import deserialize_job_hash, serialize_job_hash
from contract.shared.redis_keys import events_key, job_key
from contract.shared.serde import dumps


def now_ms() -> int:
    return int(time.time() * 1000)


class GatewayContract:
    def __init__(self) -> None:
        self._r: Optional[redis.Redis] = None
        script_dir = Path(__file__).with_name("scripts")
        self._create_job_src = script_dir.joinpath("create_job.lua").read_text(encoding="utf-8")
        self._cancel_job_src = script_dir.joinpath("cancel_job.lua").read_text(encoding="utf-8")
        self._create_job_script = None
        self._cancel_job_script = None

    async def client(self) -> redis.Redis:
        if self._r is None:
            self._r = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
            self._create_job_script = self._r.register_script(self._create_job_src)
            self._cancel_job_script = self._r.register_script(self._cancel_job_src)
        return self._r

    async def close(self) -> None:
        if self._r is not None:
            await self._r.aclose()
            self._r = None

    async def ensure_worker_group(self) -> None:
        r = await self.client()
        await ensure_worker_group(r)

    async def ping(self) -> None:
        r = await self.client()
        await r.ping()

    async def queue_stream_len(self) -> int | None:
        r = await self.client()
        try:
            return await r.xlen(shared_config.QUEUE_STREAM_KEY)
        except Exception:
            return None

    async def emit_event(
        self,
        job_id: str,
        *,
        type_: str,
        step: str,
        data: Dict[str, Any] | None,
        ttl_s: int,
    ) -> str:
        r = await self.client()
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

    async def set_job(self, job_id: str, job_obj: Dict[str, Any], ttl_s: int) -> None:
        r = await self.client()
        hk = job_key(job_id)
        mapping = serialize_job_hash(job_obj)
        await r.hset(hk, mapping=mapping)
        await r.expire(hk, ttl_s)

    async def read_job(self, job_id: str) -> Dict[str, Any]:
        r = await self.client()
        h = await r.hgetall(job_key(job_id))
        if not h:
            raise KeyError("job not found")
        return deserialize_job_hash(h)

    async def job_exists(self, job_id: str) -> bool:
        r = await self.client()
        return bool(await r.exists(job_key(job_id)))

    async def create_job(
        self,
        *,
        task: str,
        payload: Dict[str, Any],
        ttl_s: int,
        idem_key: str | None,
    ) -> Dict[str, Any]:
        r = await self.client()
        await self.ensure_worker_group()

        job_id = str(uuid.uuid4())
        created = now_ms()
        payload_json = dumps(payload)
        idem_redis_key = f"idempotency:{idem_key}" if idem_key else ""

        assert self._create_job_script is not None
        out = await self._create_job_script(
            keys=[
                idem_redis_key,
                job_key(job_id),
                events_key(job_id),
                shared_config.QUEUE_STREAM_KEY,
            ],
            args=[
                "1" if idem_key else "0",
                str(ttl_s),
                job_id,
                task,
                payload_json,
                str(created),
                "gateway.enqueue",
                dumps({}),
            ],
        )
        created_job_id = out[0]
        is_idempotent = str(out[1]) == "1"
        if is_idempotent:
            return {"job_id": created_job_id, "idempotent": True}
        return {"job_id": created_job_id}

    async def cancel_job(self, job_id: str, *, reason: Dict[str, Any] | None) -> Dict[str, Any]:
        await self.client()
        assert self._cancel_job_script is not None
        out = await self._cancel_job_script(
            keys=[job_key(job_id), events_key(job_id)],
            args=[
                str(now_ms()),
                "gateway.cancel",
                dumps(reason or {}),
                dumps({"message": "canceled", "reason": reason or {}}),
            ],
        )

        status = str(out[0])
        updated = str(out[1]) == "1"
        if status == "missing":
            raise KeyError("job not found")
        return {"job_id": job_id, "status": status, "updated": updated}

    async def get_queue_backlog(self) -> Dict[str, Any]:
        r = await self.client()
        await self.ensure_worker_group()

        pending_count = 0
        lag: Optional[int] = None

        try:
            p = await r.xpending(shared_config.QUEUE_STREAM_KEY, shared_config.WORKER_GROUP)
            if isinstance(p, dict):
                pending_count = int(p.get("pending", 0) or 0)
            elif isinstance(p, (list, tuple)) and p:
                pending_count = int(p[0] or 0)
        except Exception:
            pending_count = 0

        try:
            groups = await r.xinfo_groups(shared_config.QUEUE_STREAM_KEY)
            if isinstance(groups, list):
                for g in groups:
                    if g.get("name") == shared_config.WORKER_GROUP:
                        if "lag" in g and g["lag"] is not None:
                            lag = int(g["lag"])
                        break
        except Exception:
            lag = None

        backlog = pending_count + (lag if lag is not None else 0)
        return {"pending": pending_count, "lag": lag, "backlog": backlog}

    async def stream_events_batch(
        self, job_id: str, *, last_id: str, count: int, block_ms: int
    ) -> List[Tuple[str, Dict[str, str]]]:
        r = await self.client()
        ek = events_key(job_id)
        resp = await r.xread({ek: last_id}, count=count, block=block_ms)
        if not resp:
            return []
        _stream, entries = resp[0]
        return entries

