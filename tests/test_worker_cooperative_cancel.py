import asyncio
from dataclasses import dataclass

from worker.worker_main import process_one


@dataclass
class _Envelope:
    job_id: str
    msg_id: str
    task: str
    payload_raw: str
    ttl_s: int
    fields: dict


class _FakeClient:
    def __init__(self, *, cancel_checks: list[bool]) -> None:
        self._cancel_checks = cancel_checks
        self.acked: list[str] = []
        self.emitted = 0
        self.done = 0
        self.error = 0

    async def mark_running(self, envelope, *, step: str, consumer: str) -> bool:  # noqa: ARG002
        return True

    async def is_canceled(self, job_id: str) -> bool:  # noqa: ARG002
        if self._cancel_checks:
            return self._cancel_checks.pop(0)
        return False

    async def ack(self, msg_id: str) -> None:
        self.acked.append(msg_id)

    async def emit_message(self, envelope, *, step: str, data_obj: dict) -> None:  # noqa: ARG002
        self.emitted += 1

    async def finalize_done(self, envelope, *, step: str, result_obj: dict, started_ms: int) -> None:  # noqa: ARG002
        self.done += 1

    async def finalize_error(self, envelope, *, step: str, error_obj: dict, tb: str) -> None:  # noqa: ARG002
        self.error += 1


def test_process_one_skips_work_when_canceled_before_emit() -> None:
    client = _FakeClient(cancel_checks=[True])
    env = _Envelope(
        job_id="job-1",
        msg_id="1-0",
        task="chat",
        payload_raw='{"text":"x"}',
        ttl_s=0,
        fields={"job_id": "job-1", "task": "chat", "payload": '{"text":"x"}'},
    )

    asyncio.run(process_one(client, env))

    assert client.acked == ["1-0"]
    assert client.emitted == 0
    assert client.done == 0
    assert client.error == 0


def test_process_one_stops_after_emit_when_canceled_midflight() -> None:
    client = _FakeClient(cancel_checks=[False, True])
    env = _Envelope(
        job_id="job-2",
        msg_id="2-0",
        task="chat",
        payload_raw='{"text":"y"}',
        ttl_s=0,
        fields={"job_id": "job-2", "task": "chat", "payload": '{"text":"y"}'},
    )

    asyncio.run(process_one(client, env))

    assert client.acked == ["2-0"]
    assert client.emitted == 1
    assert client.done == 0
    assert client.error == 0
