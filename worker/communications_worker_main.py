from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
import traceback
from typing import Dict

from contract.shared import config as shared_config
from contract.worker.contract_client import ContractClient
from contract.worker.runtime import WorkerRuntime

CONSUMER = os.getenv("COMM_CONSUMER", f"communications-{os.getpid()}")
BLOCK_MS = int(os.getenv("COMM_BLOCK_MS", "5000"))
COUNT = int(os.getenv("COMM_COUNT", "10"))
MAX_INFLIGHT = int(os.getenv("COMM_MAX_INFLIGHT", "4"))
DEFAULT_TTL_S = int(os.getenv("COMM_DEFAULT_TTL_S", "3600"))
CLAIM_IDLE_MS = int(os.getenv("COMM_CLAIM_IDLE_MS", "60000"))
CLAIM_COUNT = int(os.getenv("COMM_CLAIM_COUNT", "25"))
CLAIM_EVERY_S = float(os.getenv("COMM_CLAIM_EVERY_S", "2.0"))
JOB_TIMEOUT_S = float(os.getenv("COMM_JOB_TIMEOUT_S", "0"))

_shutdown = asyncio.Event()


def now_ms() -> int:
    return int(time.time() * 1000)


def _load_payload(payload_raw: str) -> Dict[str, object]:
    try:
        obj = json.loads(payload_raw or "{}")
        if isinstance(obj, dict):
            return obj
        return {"value": obj}
    except Exception:
        return {"_raw": payload_raw}


async def process_one(client: ContractClient, envelope) -> None:
    payload_obj = _load_payload(envelope.payload_raw)
    action = str(payload_obj.get("action") or "").strip().lower()
    started = now_ms()

    ok = await client.mark_running(envelope, step="communications.start", consumer=CONSUMER)
    if not ok:
        return

    try:
        if action == "sync_communications":
            text_value = payload_obj.get("text")
            if not isinstance(text_value, str):
                raise ValueError("sync_communications requires payload.text (string)")
            await client.set_communications_text(text_value)
            await client.finalize_done(
                envelope,
                step="communications.sync.done",
                result_obj={"stored": True, "chars": len(text_value), "ms": now_ms() - started},
                started_ms=started,
            )
            return

        if action == "print_communications":
            text = await client.get_communications_text()
            if text is None:
                raise ValueError("no communications text is stored yet")
            await client.finalize_done(
                envelope,
                step="communications.print.done",
                result_obj={"text": text, "chars": len(text), "ms": now_ms() - started},
                started_ms=started,
            )
            return

        raise ValueError(f"unsupported communications action: {action or '<empty>'}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        await client.finalize_error(
            envelope,
            step="communications.error",
            error_obj={"message": str(e)},
            tb=traceback.format_exc(limit=20),
        )


async def _process_one_with_timeout(client: ContractClient, envelope) -> None:
    if JOB_TIMEOUT_S and JOB_TIMEOUT_S > 0:
        await asyncio.wait_for(process_one(client, envelope), timeout=JOB_TIMEOUT_S)
    else:
        await process_one(client, envelope)


async def pending_recovery_loop(
    client: ContractClient, sem: asyncio.Semaphore, inflight: "set[asyncio.Task]"
) -> None:
    start_id = "0-0"
    while not _shutdown.is_set():
        try:
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
    runtime = WorkerRuntime(
        default_ttl_s=DEFAULT_TTL_S,
        queue_stream_key=shared_config.COMM_QUEUE_STREAM_KEY,
        worker_group=shared_config.COMM_WORKER_GROUP,
    )
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

    recovery_task = asyncio.create_task(pending_recovery_loop(client, sem, inflight))
    try:
        while not _shutdown.is_set():
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
