from typing import Dict, List, Tuple

import redis.asyncio as redis

from contract.shared.config import QUEUE_STREAM_KEY, WORKER_GROUP
from contract.shared.group_ops import ensure_worker_group as ensure_worker_group_shared
from contract.shared.types import QueueMessage


async def ensure_worker_group(
    r: redis.Redis,
    *,
    queue_stream_key: str = QUEUE_STREAM_KEY,
    worker_group: str = WORKER_GROUP,
) -> None:
    await ensure_worker_group_shared(
        r,
        stream_key=queue_stream_key,
        group_name=worker_group,
    )


async def readgroup(
    r: redis.Redis,
    *,
    consumer: str,
    count: int,
    block_ms: int,
    queue_stream_key: str = QUEUE_STREAM_KEY,
    worker_group: str = WORKER_GROUP,
) -> List[QueueMessage]:
    resp = await r.xreadgroup(
        groupname=worker_group,
        consumername=consumer,
        streams={queue_stream_key: ">"},
        count=count,
        block=block_ms,
    )
    messages: List[QueueMessage] = []
    if not resp:
        return messages
    for _stream, entries in resp:
        for mid, mfields in entries:
            messages.append(QueueMessage(msg_id=mid, fields=mfields))
    return messages


async def autoclaim(
    r: redis.Redis,
    *,
    consumer: str,
    min_idle_ms: int,
    count: int,
    start_id: str = "0-0",
    queue_stream_key: str = QUEUE_STREAM_KEY,
    worker_group: str = WORKER_GROUP,
) -> Tuple[str, List[QueueMessage]]:
    next_start_id, messages, _deleted = await r.xautoclaim(
        name=queue_stream_key,
        groupname=worker_group,
        consumername=consumer,
        min_idle_time=min_idle_ms,
        start_id=start_id,
        count=count,
    )
    out: List[QueueMessage] = []
    for mid, mfields in messages:
        out.append(QueueMessage(msg_id=mid, fields=mfields))
    return next_start_id or "0-0", out

