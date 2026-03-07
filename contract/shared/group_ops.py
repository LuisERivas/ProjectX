from __future__ import annotations

import redis.asyncio as redis

from contract.shared import config as shared_config


async def ensure_worker_group(r: redis.Redis) -> None:
    """
    Ensure the queue stream and worker group exist.
    Uses the canonical group initialization semantics for ProjectX.
    """
    try:
        await r.xgroup_create(
            name=shared_config.QUEUE_STREAM_KEY,
            groupname=shared_config.WORKER_GROUP,
            id="$",
            mkstream=True,
        )
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            return
        raise
