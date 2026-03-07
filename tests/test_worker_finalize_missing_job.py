import asyncio
import uuid

import pytest
import redis.asyncio as redis

from contract.shared import config as shared_config
from contract.shared.redis_keys import events_key, job_key
from contract.worker.contract_client import ContractClient


def test_finalize_done_does_not_recreate_missing_job_hash(monkeypatch) -> None:
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
            pytest.skip("Redis is not available for finalize-missing-job integration test")
        await probe.aclose()

        r = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
        client = ContractClient(r, default_ttl_s=60)
        envelope = None
        try:
            await client.ensure_worker_group()
            jid = f"job-{test_suffix}"
            await r.xadd(queue_stream_key, {"job_id": jid, "task": "chat", "payload": "{}"})
            envelopes = await client.readgroup(consumer="worker-test", count=1, block_ms=1)
            assert len(envelopes) == 1
            envelope = envelopes[0]

            # Simulate key expiry/deletion before finalize path executes.
            await r.delete(job_key(jid))
            await r.delete(events_key(jid))

            await client.finalize_done(
                envelope,
                step="worker.done",
                result_obj={"ok": True},
                started_ms=1,
            )

            assert await r.hgetall(job_key(jid)) == {}
            assert await r.xrange(events_key(jid), "-", "+") == []

            pending = await r.xpending(queue_stream_key, worker_group)
            if isinstance(pending, dict):
                assert int(pending.get("pending", 0) or 0) == 0
            elif isinstance(pending, (list, tuple)) and pending:
                assert int(pending[0] or 0) == 0
        finally:
            try:
                await r.delete(queue_stream_key)
                if envelope is not None:
                    await r.delete(job_key(envelope.job_id))
                    await r.delete(events_key(envelope.job_id))
            finally:
                await r.aclose()

    asyncio.run(scenario())
