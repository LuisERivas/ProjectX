from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4


DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_POLL_TIMEOUT_S = 45.0
POLL_INTERVAL_S = 1.0
TERMINAL_STATUSES = {"done", "error", "canceled"}


@dataclass
class HttpResult:
    status: int
    body_text: str
    data: Any


def _normalize_host(host: str) -> str:
    host = (host or "").strip()
    if host.startswith("http://"):
        host = host[len("http://") :]
    elif host.startswith("https://"):
        host = host[len("https://") :]
    return host.strip("/")


def _request_json(
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> HttpResult:
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    data: Optional[bytes] = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, method=method, headers=req_headers)
    body_text = ""
    status = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            body_bytes = resp.read()
            body_text = body_bytes.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        status = int(e.code)
        body_text = e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"connection failed: {e.reason}") from e

    parsed: Any = body_text
    try:
        parsed = json.loads(body_text) if body_text else {}
    except Exception:
        pass
    return HttpResult(status=status, body_text=body_text, data=parsed)


def _expect(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def _step(name: str) -> None:
    print(f"\n== {name} ==")


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _wait_for_terminal(base_url: str, job_id: str, poll_timeout_s: float) -> Dict[str, Any]:
    deadline = time.time() + poll_timeout_s
    last_status = ""
    last_body = ""
    while time.time() < deadline:
        resp = _request_json("GET", f"{base_url}/v1/jobs/{job_id}")
        _expect(resp.status == 200, f"GET /v1/jobs/{job_id} returned {resp.status}: {resp.body_text}")
        _expect(isinstance(resp.data, dict), "job response is not a JSON object")
        status = str(resp.data.get("status", "")).strip()
        last_status = status
        last_body = resp.body_text
        if status in TERMINAL_STATUSES:
            return resp.data
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(
        f"job {job_id} did not reach terminal status within {poll_timeout_s:.0f}s "
        f"(last status={last_status!r}, body={last_body})"
    )


def _run_automated_tests(base_url: str, timeout_s: float, poll_timeout_s: float) -> None:
    _step("Health check")
    health = _request_json("GET", f"{base_url}/health", timeout_s=timeout_s)
    _expect(health.status == 200, f"GET /health returned {health.status}: {health.body_text}")
    _expect(isinstance(health.data, dict), "health response is not JSON")
    _expect(bool(health.data.get("ok")) is True, "health response does not include ok=true")
    _ok("GET /health is reachable and reports ok=true")

    _step("Submit + read terminal state")
    submit = _request_json(
        "POST",
        f"{base_url}/v1/jobs",
        payload={"task": "chat", "payload": {"text": "hello from remote test runner"}},
        timeout_s=timeout_s,
    )
    _expect(submit.status == 200, f"POST /v1/jobs returned {submit.status}: {submit.body_text}")
    _expect(isinstance(submit.data, dict), "submit response is not JSON")
    job_id = str(submit.data.get("job_id", "")).strip()
    _expect(job_id, "submit response missing job_id")
    _ok(f"submitted job_id={job_id}")

    final_job = _wait_for_terminal(base_url, job_id, poll_timeout_s)
    _expect(str(final_job.get("status", "")).strip() in TERMINAL_STATUSES, "job did not reach terminal state")
    _ok(f"job reached terminal status={final_job.get('status')}")

    _step("Idempotency")
    idem_key = f"remote-test-{uuid4().hex}"
    idem_payload = {"task": "chat", "payload": {"text": "idempotency check"}}
    idem_1 = _request_json(
        "POST",
        f"{base_url}/v1/jobs",
        payload=idem_payload,
        headers={"Idempotency-Key": idem_key},
        timeout_s=timeout_s,
    )
    idem_2 = _request_json(
        "POST",
        f"{base_url}/v1/jobs",
        payload=idem_payload,
        headers={"Idempotency-Key": idem_key},
        timeout_s=timeout_s,
    )
    _expect(idem_1.status == 200, f"first idempotent POST failed: {idem_1.status}: {idem_1.body_text}")
    _expect(idem_2.status == 200, f"second idempotent POST failed: {idem_2.status}: {idem_2.body_text}")
    _expect(isinstance(idem_1.data, dict) and isinstance(idem_2.data, dict), "idempotency responses are not JSON")
    id1 = str(idem_1.data.get("job_id", "")).strip()
    id2 = str(idem_2.data.get("job_id", "")).strip()
    _expect(id1 and id2, "idempotency response missing job_id")
    _expect(id1 == id2, f"idempotency mismatch: first={id1}, second={id2}")
    _expect(bool(idem_2.data.get("idempotent")) is True, "second idempotent call did not return idempotent=true")
    _ok("idempotency returns same job_id and idempotent=true on second call")

    _step("Cancel behavior")
    cancel_submit = _request_json(
        "POST",
        f"{base_url}/v1/jobs",
        payload={"task": "chat", "payload": {"text": "cancel check"}},
        timeout_s=timeout_s,
    )
    _expect(cancel_submit.status == 200, f"cancel submit failed: {cancel_submit.status}: {cancel_submit.body_text}")
    _expect(isinstance(cancel_submit.data, dict), "cancel submit response is not JSON")
    cancel_job_id = str(cancel_submit.data.get("job_id", "")).strip()
    _expect(cancel_job_id, "cancel submit missing job_id")

    cancel_resp = _request_json(
        "POST",
        f"{base_url}/v1/jobs/{cancel_job_id}/cancel",
        payload={"reason": {"by": "remote-test-runner"}},
        timeout_s=timeout_s,
    )
    _expect(
        cancel_resp.status == 200,
        f"cancel endpoint failed: {cancel_resp.status}: {cancel_resp.body_text}",
    )
    _expect(isinstance(cancel_resp.data, dict), "cancel response is not JSON")
    cancel_status = str(cancel_resp.data.get("status", "")).strip()
    _expect(cancel_status in TERMINAL_STATUSES, f"unexpected cancel status: {cancel_status!r}")
    _ok(f"cancel endpoint reachable; returned status={cancel_status}")


def _print_manual_followups(host: str) -> None:
    print("\n== Manual follow-up checks (run on Jetson shell) ==")
    print("These checks from TESTING.md require local Jetson access (no SSH mode).")
    print("")
    print("# Service status checks")
    print("sudo systemctl status redis")
    print("sudo systemctl status redis-gateway.service")
    print("sudo systemctl status redis-echo-worker.service")
    print("")
    print("# Toolchain checks")
    print("g++ --version")
    print("cmake --version")
    print("")
    print("# Redis ground-truth checks")
    print('JOB_ID="<paste-job-id-from-runner-output>"')
    print('redis-cli -u redis://127.0.0.1:6379/0 HGETALL "job:${JOB_ID}"')
    print('redis-cli -u redis://127.0.0.1:6379/0 XRANGE "job:${JOB_ID}:events" - +')
    print('redis-cli -u redis://127.0.0.1:6379/0 XPENDING "jobs:stream" "workers"')
    print("")
    print("# Boundary static test")
    print("cd /home/jetson/ProjectX")
    print("python -m pytest tests/test_boundary_rules.py")
    print("")
    print(f"# Remote health sanity check from this machine")
    print(f"curl -sS http://{host}:8000/health")


def _resolve_host(args: argparse.Namespace) -> str:
    if args.host:
        host = _normalize_host(args.host)
    else:
        host = _normalize_host(input("Enter Jetson IP address (or host): ").strip())
    if not host:
        raise RuntimeError("host/IP is required")
    return host


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run remote, non-disruptive checks aligned with TESTING.md. "
            "Prompts for Jetson host/IP unless --host is provided."
        )
    )
    parser.add_argument("--host", help="Jetson host or IP (for example: 192.168.1.50)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Gateway port (default: 8000)")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-request timeout in seconds (default: 8)",
    )
    parser.add_argument(
        "--poll-timeout-s",
        type=float,
        default=DEFAULT_POLL_TIMEOUT_S,
        help="Max wait for terminal job status in seconds (default: 45)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        host = _resolve_host(args)
        base_url = f"http://{host}:{args.port}"
        print(f"Target base URL: {base_url}")
        _run_automated_tests(base_url, timeout_s=args.timeout_s, poll_timeout_s=args.poll_timeout_s)
        _print_manual_followups(host)
        print("\nResult: ALL automated checks passed.")
        return 0
    except Exception as e:
        _fail(str(e))
        print("Result: one or more automated checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
