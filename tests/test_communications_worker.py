import asyncio
from dataclasses import dataclass

from worker.communications_worker_main import process_one


@dataclass
class _Envelope:
    job_id: str
    msg_id: str
    task: str
    payload_raw: str
    ttl_s: int
    fields: dict


class _FakeClient:
    def __init__(self) -> None:
        self.text_store: str | None = None
        self.done_calls: list[dict] = []
        self.error_calls: list[dict] = []

    async def mark_running(self, envelope, *, step: str, consumer: str) -> bool:  # noqa: ARG002
        return True

    async def set_communications_text(self, text: str) -> None:
        self.text_store = text

    async def get_communications_text(self) -> str | None:
        return self.text_store

    async def finalize_done(self, envelope, *, step: str, result_obj: dict, started_ms: int) -> None:  # noqa: ARG002
        self.done_calls.append({"step": step, "result": result_obj})

    async def finalize_error(self, envelope, *, step: str, error_obj: dict, tb: str) -> None:  # noqa: ARG002
        self.error_calls.append({"step": step, "error": error_obj, "tb": tb})


def test_sync_action_stores_text_and_finalizes_done() -> None:
    client = _FakeClient()
    env = _Envelope(
        job_id="job-sync",
        msg_id="1-0",
        task="tool",
        payload_raw='{"action":"sync_communications","text":"alpha"}',
        ttl_s=0,
        fields={"job_id": "job-sync", "task": "tool"},
    )

    asyncio.run(process_one(client, env))

    assert client.text_store == "alpha"
    assert len(client.done_calls) == 1
    assert client.done_calls[0]["step"] == "communications.sync.done"
    assert client.error_calls == []


def test_print_action_reads_text_and_finalizes_done() -> None:
    client = _FakeClient()
    client.text_store = "beta"
    env = _Envelope(
        job_id="job-print",
        msg_id="2-0",
        task="tool",
        payload_raw='{"action":"print_communications"}',
        ttl_s=0,
        fields={"job_id": "job-print", "task": "tool"},
    )

    asyncio.run(process_one(client, env))

    assert len(client.done_calls) == 1
    assert client.done_calls[0]["step"] == "communications.print.done"
    assert client.done_calls[0]["result"]["text"] == "beta"
    assert client.error_calls == []


def test_print_action_errors_when_missing_text() -> None:
    client = _FakeClient()
    env = _Envelope(
        job_id="job-print-missing",
        msg_id="3-0",
        task="tool",
        payload_raw='{"action":"print_communications"}',
        ttl_s=0,
        fields={"job_id": "job-print-missing", "task": "tool"},
    )

    asyncio.run(process_one(client, env))

    assert client.done_calls == []
    assert len(client.error_calls) == 1
    assert client.error_calls[0]["step"] == "communications.error"
