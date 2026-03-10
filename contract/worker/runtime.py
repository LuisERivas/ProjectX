from __future__ import annotations

from typing import Optional

import redis.asyncio as redis

from contract.shared import config as shared_config
from contract.worker.contract_client import ContractClient


class WorkerRuntime:
    def __init__(
        self,
        *,
        default_ttl_s: int,
        queue_stream_key: str = shared_config.QUEUE_STREAM_KEY,
        worker_group: str = shared_config.WORKER_GROUP,
    ) -> None:
        self._default_ttl_s = default_ttl_s
        self._queue_stream_key = queue_stream_key
        self._worker_group = worker_group
        self._redis: Optional[redis.Redis] = None
        self.client: Optional[ContractClient] = None

    async def start(self) -> ContractClient:
        self._redis = redis.from_url(shared_config.REDIS_URL, decode_responses=True)
        await self._redis.ping()
        self.client = ContractClient(
            self._redis,
            default_ttl_s=self._default_ttl_s,
            queue_stream_key=self._queue_stream_key,
            worker_group=self._worker_group,
        )
        await self.client.ensure_worker_group()
        return self.client

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        self.client = None

