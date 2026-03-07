from typing import Dict

import redis.asyncio as redis

from contract.shared.redis_keys import dlq_stream_key
from contract.shared.serde import dumps


async def push_dlq(r: redis.Redis, fields: Dict[str, str]) -> None:
    await r.xadd(dlq_stream_key(), fields)

