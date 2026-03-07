import asyncio
import uuid

import pytest
import redis.asyncio as redis

from contract.gateway.service import GatewayContract
from contract.shared import config as shared_config
from contract.shared.redis_keys import events_key, job_key


def test_create_job_idempotency_is_atomic(monkeypatch) -> None:
    test_suffix = uuid.uuid4().hex
    queue_stream_key = f"test:jobs:stream:{test_suffix}"
    worker_group = f"test:workers:{test_suffix}"
    idem_key = f"test-idem-{test_suffix}"

    monkeypatch.setattr(shared_config, "QUEUE_STREAM_KEY", queue_stream_key, raising=False)
    monkeypatch.setattr(shared_config, "WORKER_GROUP", worker_group, raising=False)

    async def scenario() -> None:
        probe = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
        try:
            await probe.ping()
        except Exception:
            await probe.aclose()
            pytest.skip("Redis is not available for idempotency integration test")

        await probe.aclose()

        gateway = GatewayContract()
        created_job_id = ""
        try:
            async def submit() -> dict:
                return await gateway.create_job(
                    task="chat",
                    payload={"text": "idempotency-concurrency-test"},
                    ttl_s=60,
                    idem_key=idem_key,
                )

            # Fire concurrent submits with the same idempotency key.
            results = await asyncio.gather(*[submit() for _ in range(20)])
            job_ids = {item["job_id"] for item in results}
            assert len(job_ids) == 1

            created_job_id = next(iter(job_ids))
            idempotent_count = sum(1 for item in results if item.get("idempotent"))
            assert idempotent_count == len(results) - 1

            r = await gateway.client()
            queue_entries = await r.xrange(queue_stream_key, "-", "+")
            matching_queue_entries = [
                (entry_id, fields)
                for entry_id, fields in queue_entries
                if fields.get("job_id") == created_job_id
            ]
            assert len(matching_queue_entries) == 1

            job_hash = await r.hgetall(job_key(created_job_id))
            assert job_hash.get("status") == "queued"

            queued_events = await r.xrange(events_key(created_job_id), "-", "+")
            queued_count = sum(1 for _, fields in queued_events if fields.get("type") == "queued")
            assert queued_count == 1
        finally:
            try:
                r = await gateway.client()
                if created_job_id:
                    await r.delete(job_key(created_job_id))
                    await r.delete(events_key(created_job_id))
                await r.delete(queue_stream_key)
                await r.delete(f"idempotency:{idem_key}")
            finally:
                await gateway.close()

    asyncio.run(scenario())
