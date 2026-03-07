from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
import traceback
from typing import Any, Dict

from contract.worker.contract_client import ContractClient
from contract.worker.runtime import WorkerRuntime

# --- worker runtime tuning ---
CONSUMER = os.getenv("CONSUMER", f"echo-{os.getpid()}")
BLOCK_MS = int(os.getenv("BLOCK_MS", "5000"))
COUNT = int(os.getenv("COUNT", "10"))
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "4"))
DEFAULT_TTL_S = int(os.getenv("DEFAULT_TTL_S", "3600"))

# --- reliability hardening ---
# Reclaim messages that have been idle (unacked) for this long.
CLAIM_IDLE_MS = int(os.getenv("CLAIM_IDLE_MS", "60000"))  # 60s
# How many pending messages to attempt to reclaim per pass.
CLAIM_COUNT = int(os.getenv("CLAIM_COUNT", "25"))
# How often to run the pending recovery loop.
CLAIM_EVERY_S = float(os.getenv("CLAIM_EVERY_S", "2.0"))

# Per-job timeout (0 disables). This is wall-clock time for processing_one().
JOB_TIMEOUT_S = float(os.getenv("JOB_TIMEOUT_S", "0"))

_shutdown = asyncio.Event()


def now_ms() -> int:
    return int(time.time() * 1000)


async def process_one(client: ContractClient, envelope) -> None:
    """
    Queue message fields: job_id, task, payload (payload is JSON string).
    Updates job hash + job events. ACK last.
    """
    msg_fields: Dict[str, str] = envelope.fields
    task = (msg_fields.get("task") or "").strip()

    try:
        payload_obj = json.loads(envelope.payload_raw or "{}")
        if not isinstance(payload_obj, dict):
            payload_obj = {"value": payload_obj}
    except Exception:
        payload_obj = {"_raw": envelope.payload_raw}

    started = now_ms()

    ok = await client.mark_running(envelope, step="worker.start", consumer=CONSUMER)
    if not ok:
        return

    try:
        # Cooperative cancel check: do not emit post-cancel events/results.
        if await client.is_canceled(envelope.job_id):
            await client.ack(envelope.msg_id)
            return

        # --- echo logic (placeholder; replace later with local model call) ---
        result_text = f"echo(task={task}): {payload_obj}"

        await client.emit_message(
            envelope,
            step="worker.echo",
            data_obj={"text": result_text},
        )

        if await client.is_canceled(envelope.job_id):
            await client.ack(envelope.msg_id)
            return

        await client.finalize_done(
            envelope,
            step="worker.done",
            result_obj={"text": result_text, "ms": now_ms() - started},
            started_ms=started,
        )

    except asyncio.CancelledError:
        # Best-effort: do not ACK; allow claim/retry by another consumer.
        raise
    except Exception as e:
        if await client.is_canceled(envelope.job_id):
            await client.ack(envelope.msg_id)
            return
        err_msg = str(e)
        tb = traceback.format_exc(limit=20)
        await client.finalize_error(
            envelope,
            step="worker.error",
            error_obj={"message": err_msg},
            tb=tb,
        )


async def _process_one_with_timeout(
    client: ContractClient, envelope
) -> None:
    if JOB_TIMEOUT_S and JOB_TIMEOUT_S > 0:
        await asyncio.wait_for(process_one(client, envelope), timeout=JOB_TIMEOUT_S)
    else:
        await process_one(client, envelope)


async def pending_recovery_loop(
    client: ContractClient, sem: asyncio.Semaphore, inflight: "set[asyncio.Task]"
) -> None:
    """
    Reclaim stuck pending messages (e.g., worker crash after delivery, before ACK).
    Uses XAUTOCLAIM to transfer ownership to this consumer, then processes them.
    """
    start_id = "0-0"
    while not _shutdown.is_set():
        try:
            # Don't reclaim if we have no capacity.
            if sem.locked():
                await asyncio.sleep(CLAIM_EVERY_S)
                continue

            next_start_id, envelopes = await client.autoclaim(
                consumer=CONSUMER,
                min_idle_ms=CLAIM_IDLE_MS,
                count=CLAIM_COUNT,
                start_id=start_id,
            )
            start_id = next_start_id or "0-0"

            if not envelopes:
                await asyncio.sleep(CLAIM_EVERY_S)
                continue

            for env in envelopes:
                # Gate on semaphore; ensures we don't exceed MAX_INFLIGHT
                await sem.acquire()

                async def _run(envelope_) -> None:
                    try:
                        await _process_one_with_timeout(client, envelope_)
                    finally:
                        sem.release()

                t = asyncio.create_task(_run(env))
                inflight.add(t)
                t.add_done_callback(lambda tt: inflight.discard(tt))

        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(CLAIM_EVERY_S)


async def run() -> None:
    runtime = WorkerRuntime(default_ttl_s=DEFAULT_TTL_S)
    client = await runtime.start()

    sem = asyncio.Semaphore(MAX_INFLIGHT)
    inflight: set[asyncio.Task] = set()

    async def spawn(envelope) -> None:
        await sem.acquire()

        async def _run() -> None:
            try:
                await _process_one_with_timeout(client, envelope)
            finally:
                sem.release()

        t = asyncio.create_task(_run())
        inflight.add(t)
        t.add_done_callback(lambda tt: inflight.discard(tt))

    # Start pending recovery loop
    recovery_task = asyncio.create_task(pending_recovery_loop(client, sem, inflight))

    try:
        while not _shutdown.is_set():
            try:
                # If we are saturated, wait until capacity returns before reading more.
                if sem.locked():
                    await asyncio.sleep(0.01)
                    continue

                envelopes = await client.readgroup(
                    consumer=CONSUMER,
                    count=COUNT,
                    block_ms=BLOCK_MS,
                )
                if not envelopes:
                    continue

                for env in envelopes:
                    await spawn(env)

            except asyncio.CancelledError:
                break
            except Exception:
                # brief backoff on transient errors
                await asyncio.sleep(0.25)

    finally:
        _shutdown.set()
        recovery_task.cancel()
        with contextlib.suppress(Exception):
            await recovery_task

        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        await runtime.close()


def _signal_handler(*_args: object) -> None:
    _shutdown.set()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    asyncio.run(run())

