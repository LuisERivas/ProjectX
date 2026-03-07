import asyncio
import uuid

import pytest
import redis.asyncio as redis

from contract.gateway.service import GatewayContract
from contract.shared import config as shared_config
from contract.shared.redis_keys import events_key, job_key


def test_cancel_job_marks_terminal_and_is_idempotent(monkeypatch) -> None:
    test_suffix = uuid.uuid4().hex
    queue_stream_key = f"test:jobs:stream:{test_suffix}"
    worker_group = f"test:workers:{test_suffix}"

    monkeypatch.setattr(shared_config, "QUEUE_STREAM_KEY", queue_stream_key, raising=False)
    monkeypatch.setattr(shared_config, "WORKER_GROUP", worker_group, raising=False)

    async def scenario() -> None:
        probe = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
        try:
            await probe.ping()
        except Exception:
            await probe.aclose()
            pytest.skip("Redis is not available for cancel integration test")
        await probe.aclose()

        gateway = GatewayContract()
        created_job_id = ""
        try:
            created = await gateway.create_job(
                task="chat",
                payload={"text": "cancel-me"},
                ttl_s=60,
                idem_key=None,
            )
            created_job_id = created["job_id"]

            first = await gateway.cancel_job(created_job_id, reason={"by": "test"})
            assert first["status"] == "canceled"
            assert first["updated"] is True

            second = await gateway.cancel_job(created_job_id, reason={"by": "test"})
            assert second["status"] == "canceled"
            assert second["updated"] is False

            r = await gateway.client()
            job_hash = await r.hgetall(job_key(created_job_id))
            assert job_hash.get("status") == "canceled"

            evs = await r.xrange(events_key(created_job_id), "-", "+")
            canceled_count = sum(1 for _, fields in evs if fields.get("type") == "canceled")
            assert canceled_count == 1
        finally:
            try:
                if created_job_id:
                    r = await gateway.client()
                    await r.delete(job_key(created_job_id))
                    await r.delete(events_key(created_job_id))
                r = await gateway.client()
                await r.delete(queue_stream_key)
            finally:
                await gateway.close()

    asyncio.run(scenario())
