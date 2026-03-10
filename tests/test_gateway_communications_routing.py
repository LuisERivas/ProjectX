import asyncio
import uuid

import pytest
import redis.asyncio as redis

from contract.gateway.service import GatewayContract
from contract.shared import config as shared_config
from contract.shared.redis_keys import job_key


def test_gateway_routes_communications_jobs_to_dedicated_stream(monkeypatch) -> None:
    suffix = uuid.uuid4().hex
    queue_stream_key = f"test:jobs:stream:{suffix}"
    worker_group = f"test:workers:{suffix}"
    comm_stream_key = f"test:jobs:communications:stream:{suffix}"
    comm_group = f"test:communications-workers:{suffix}"

    monkeypatch.setattr(shared_config, "QUEUE_STREAM_KEY", queue_stream_key, raising=False)
    monkeypatch.setattr(shared_config, "WORKER_GROUP", worker_group, raising=False)
    monkeypatch.setattr(shared_config, "COMM_QUEUE_STREAM_KEY", comm_stream_key, raising=False)
    monkeypatch.setattr(shared_config, "COMM_WORKER_GROUP", comm_group, raising=False)

    async def scenario() -> None:
        probe = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
        try:
            await probe.ping()
        except Exception:
            await probe.aclose()
            pytest.skip("Redis is not available for routing integration test")
        await probe.aclose()

        gateway = GatewayContract()
        r = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
        tool_job_id = ""
        chat_job_id = ""
        try:
            tool_job = await gateway.create_job(
                task="tool",
                payload={"action": "sync_communications", "text": "hello"},
                ttl_s=60,
                idem_key=None,
            )
            tool_job_id = tool_job["job_id"]

            chat_job = await gateway.create_job(
                task="chat",
                payload={"text": "normal-flow"},
                ttl_s=60,
                idem_key=None,
            )
            chat_job_id = chat_job["job_id"]

            comm_entries = await r.xrange(comm_stream_key, "-", "+")
            default_entries = await r.xrange(queue_stream_key, "-", "+")

            comm_job_ids = {fields.get("job_id") for _, fields in comm_entries}
            default_job_ids = {fields.get("job_id") for _, fields in default_entries}

            assert tool_job_id in comm_job_ids
            assert tool_job_id not in default_job_ids
            assert chat_job_id in default_job_ids
            assert chat_job_id not in comm_job_ids
        finally:
            if tool_job_id:
                await r.delete(job_key(tool_job_id))
                await r.delete(f"job:{tool_job_id}:events")
            if chat_job_id:
                await r.delete(job_key(chat_job_id))
                await r.delete(f"job:{chat_job_id}:events")
            await r.delete(comm_stream_key)
            await r.delete(queue_stream_key)
            await gateway.close()
            await r.aclose()

    asyncio.run(scenario())
