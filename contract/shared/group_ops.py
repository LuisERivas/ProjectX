from __future__ import annotations

import redis.asyncio as redis

from contract.shared import config as shared_config


async def ensure_worker_group(
    r: redis.Redis,
    *,
    stream_key: str | None = None,
    group_name: str | None = None,
) -> None:
    """
    Ensure the queue stream and worker group exist.
    Uses the canonical group initialization semantics for ProjectX.
    """
    queue_key = stream_key or shared_config.QUEUE_STREAM_KEY
    worker_group = group_name or shared_config.WORKER_GROUP
    try:
        await r.xgroup_create(
            name=queue_key,
            groupname=worker_group,
            id="$",
            mkstream=True,
        )
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            return
        raise
