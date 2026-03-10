from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import redis.asyncio as redis

from contract.shared import config as shared_config
from contract.shared.redis_keys import communications_text_key, events_key, job_key
from contract.shared.schema import validate_queue_message
from contract.shared.serde import dumps
from contract.shared.types import JobEnvelope, QueueMessage
from contract.worker.dlq import push_dlq
from contract.worker.errors import ContractViolation
from contract.worker.stream_ops import autoclaim, ensure_worker_group, readgroup


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


class ContractClient:
    def __init__(
        self,
        r: redis.Redis,
        *,
        default_ttl_s: int,
        queue_stream_key: str = shared_config.QUEUE_STREAM_KEY,
        worker_group: str = shared_config.WORKER_GROUP,
    ) -> None:
        self._r = r
        self._default_ttl_s = default_ttl_s
        self._queue_stream_key = queue_stream_key
        self._worker_group = worker_group
        scripts_dir = Path(__file__).with_name("scripts")
        self._finalize_and_ack = self._r.register_script(
            scripts_dir.joinpath("finalize_and_ack.lua").read_text(encoding="utf-8")
        )
        self._mark_running = self._r.register_script(
            scripts_dir.joinpath("mark_running.lua").read_text(encoding="utf-8")
        )

    async def readgroup(self, *, consumer: str, count: int, block_ms: int) -> List[JobEnvelope]:
        msgs = await readgroup(
            self._r,
            consumer=consumer,
            count=count,
            block_ms=block_ms,
            queue_stream_key=self._queue_stream_key,
            worker_group=self._worker_group,
        )
        return [self._to_envelope(m) for m in msgs]

    async def autoclaim(
        self, *, consumer: str, min_idle_ms: int, count: int, start_id: str = "0-0"
    ) -> Tuple[str, List[JobEnvelope]]:
        next_id, msgs = await autoclaim(
            self._r,
            consumer=consumer,
            min_idle_ms=min_idle_ms,
            count=count,
            start_id=start_id,
            queue_stream_key=self._queue_stream_key,
            worker_group=self._worker_group,
        )
        return next_id, [self._to_envelope(m) for m in msgs]

    def _to_envelope(self, msg: QueueMessage) -> JobEnvelope:
        fields = msg.fields
        validate_queue_message(fields)
        job_id = (fields.get("job_id") or "").strip()
        task = (fields.get("task") or "").strip()
        payload_raw = fields.get("payload") or "{}"
        if not job_id:
            raise ContractViolation("missing job_id")
        return JobEnvelope(
            job_id=job_id,
            msg_id=msg.msg_id,
            task=task,
            payload_raw=payload_raw,
            ttl_s=0,
            fields=dict(fields),
        )

    async def ensure_worker_group(self) -> None:
        await ensure_worker_group(
            self._r,
            queue_stream_key=self._queue_stream_key,
            worker_group=self._worker_group,
        )

    async def status(self, job_id: str) -> str:
        raw = await self._r.hget(job_key(job_id), "status")
        return (raw or "").strip()

    async def is_canceled(self, job_id: str) -> bool:
        return await self.status(job_id) == "canceled"

    async def ack(self, msg_id: str) -> None:
        await self._r.xack(self._queue_stream_key, self._worker_group, msg_id)

    async def mark_running(self, envelope: JobEnvelope, *, step: str, consumer: str) -> bool:
        job_id = envelope.job_id
        started = _now_ms()
        out = await self._mark_running(
            keys=[job_key(job_id), events_key(job_id), self._queue_stream_key],
            args=[
                self._worker_group,
                envelope.msg_id,
                str(self._default_ttl_s),
                str(started),
                step,
                dumps({"consumer": consumer, "task": envelope.task}),
            ],
        )
        should_run = str(out[0]) == "run"
        if not should_run:
            return False
        ttl_raw = str(out[1]) if len(out) > 1 else ""
        try:
            envelope.ttl_s = int(ttl_raw)
        except Exception:
            envelope.ttl_s = self._default_ttl_s
        return True

    async def emit_message(self, envelope: JobEnvelope, *, step: str, data_obj: Dict[str, Any]) -> None:
        job_id = envelope.job_id
        ttl_s = envelope.ttl_s or self._default_ttl_s
        now = _now_ms()
        ek = events_key(job_id)
        fields = {
            "type": "message",
            "ts": str(now),
            "step": step,
            "data": dumps(data_obj),
        }
        pipe = self._r.pipeline()
        pipe.xadd(ek, fields)
        pipe.expire(ek, ttl_s)
        await pipe.execute()

    async def finalize_done(
        self,
        envelope: JobEnvelope,
        *,
        step: str,
        result_obj: Dict[str, Any],
        started_ms: int,
    ) -> None:
        job_id = envelope.job_id
        ttl_s = envelope.ttl_s or self._default_ttl_s
        done_ts = _now_ms()
        duration_ms = done_ts - started_ms

        await self._finalize_and_ack(
            keys=[job_key(job_id), events_key(job_id), self._queue_stream_key],
            args=[
                self._worker_group,
                envelope.msg_id,
                str(ttl_s),
                "done",
                str(done_ts),
                step,
                dumps(result_obj),
                "",
                dumps({"ms": duration_ms}),
            ],
        )

    async def finalize_error(
        self,
        envelope: JobEnvelope,
        *,
        step: str,
        error_obj: Dict[str, Any],
        tb: str,
    ) -> None:
        job_id = envelope.job_id
        ttl_s = envelope.ttl_s or self._default_ttl_s
        err_ts = _now_ms()
        dlq_fields: Dict[str, str] = {
            "src_stream": self._queue_stream_key,
            "src_group": self._worker_group,
            "src_msg_id": envelope.msg_id,
            "consumer": "",
            "ts": str(err_ts),
            "error": str(error_obj.get("message", "")),
            "traceback": tb,
            "job_id": job_id,
            "task": envelope.task,
            "payload": envelope.payload_raw,
            "raw": dumps(envelope.fields),
        }
        await push_dlq(self._r, dlq_fields)

        await self._finalize_and_ack(
            keys=[job_key(job_id), events_key(job_id), self._queue_stream_key],
            args=[
                self._worker_group,
                envelope.msg_id,
                str(ttl_s),
                "error",
                str(err_ts),
                step,
                "",
                dumps(error_obj),
                dumps(error_obj),
            ],
        )

    async def set_communications_text(self, text: str) -> None:
        await self._r.set(communications_text_key(), text)

    async def get_communications_text(self) -> str | None:
        raw = await self._r.get(communications_text_key())
        if raw is None:
            return None
        return str(raw)

