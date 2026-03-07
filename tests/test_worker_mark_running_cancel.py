import asyncio
import uuid

import pytest
import redis.asyncio as redis

from contract.gateway.service import GatewayContract
from contract.shared import config as shared_config
from contract.shared.redis_keys import events_key, job_key
from contract.worker.contract_client import ContractClient


def test_mark_running_does_not_overwrite_canceled(monkeypatch) -> None:
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
            pytest.skip("Redis is not available for worker cancel race integration test")
        await probe.aclose()

        gateway = GatewayContract()
        worker_redis = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
        client = ContractClient(worker_redis, default_ttl_s=60)
        created_job_id = ""
        try:
            created = await gateway.create_job(
                task="chat",
                payload={"text": "cancel-race"},
                ttl_s=60,
                idem_key=None,
            )
            created_job_id = created["job_id"]

            envelopes = await client.readgroup(consumer="worker-test", count=1, block_ms=1)
            assert len(envelopes) == 1
            envelope = envelopes[0]

            canceled = await gateway.cancel_job(created_job_id, reason={"by": "test"})
            assert canceled["status"] == "canceled"

            should_run = await client.mark_running(
                envelope,
                step="worker.start",
                consumer="worker-test",
            )
            assert should_run is False

            job_hash = await worker_redis.hgetall(job_key(created_job_id))
            assert job_hash.get("status") == "canceled"

            evs = await worker_redis.xrange(events_key(created_job_id), "-", "+")
            canceled_count = sum(1 for _, fields in evs if fields.get("type") == "canceled")
            assert canceled_count == 1
        finally:
            try:
                if created_job_id:
                    await worker_redis.delete(job_key(created_job_id))
                    await worker_redis.delete(events_key(created_job_id))
                await worker_redis.delete(queue_stream_key)
            finally:
                await gateway.close()
                await worker_redis.aclose()

    asyncio.run(scenario())
