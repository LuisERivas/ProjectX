from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict
from urllib import error, request


TERMINAL_STATUSES = {"done", "error", "canceled"}


def _http_json(method: str, url: str, payload: Dict[str, Any] | None, timeout_s: float) -> Dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, method=method.upper(), data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else {}
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {e.code} {detail}") from e


def _wait_for_terminal(base_url: str, job_id: str, timeout_s: float, poll_s: float) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    url = f"{base_url}/v1/jobs/{job_id}"
    while time.time() < deadline:
        job = _http_json("GET", url, payload=None, timeout_s=timeout_s)
        status = str(job.get("status") or "")
        if status in TERMINAL_STATUSES:
            return job
        time.sleep(poll_s)
    raise RuntimeError(f"timed out waiting for terminal status for job_id={job_id}")


def _create_job(base_url: str, payload: Dict[str, Any], timeout_s: float) -> str:
    out = _http_json(
        "POST",
        f"{base_url}/v1/jobs",
        payload=payload,
        timeout_s=timeout_s,
    )
    job_id = str(out.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"gateway response missing job_id: {out}")
    return job_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync/print communications.txt through ProjectX gateway jobs.")
    parser.add_argument("--host", default="127.0.0.1", help="Gateway host")
    parser.add_argument("--port", type=int, default=8000, help="Gateway port")
    parser.add_argument("--timeout-s", type=float, default=45.0, help="Request/poll timeout seconds")
    parser.add_argument("--poll-s", type=float, default=0.5, help="Poll interval seconds")
    parser.add_argument(
        "--worker-file",
        default="communications.txt",
        help="File path (on worker host project root) to read before syncing to Redis.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip sync step and only request print from Redis.",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if not args.skip_sync:
        sync_job_id = _create_job(
            base_url,
            payload={
                "task": "tool",
                "payload": {
                    "action": "sync_communications_from_file",
                    "file_path": args.worker_file,
                },
            },
            timeout_s=args.timeout_s,
        )
        sync_terminal = _wait_for_terminal(base_url, sync_job_id, timeout_s=args.timeout_s, poll_s=args.poll_s)
        if sync_terminal.get("status") != "done":
            raise SystemExit(f"sync job failed: {sync_terminal}")
        print(f"sync done: job_id={sync_job_id}")

    print_job_id = _create_job(
        base_url,
        payload={
            "task": "tool",
            "payload": {
                "action": "print_communications",
            },
        },
        timeout_s=args.timeout_s,
    )
    print_terminal = _wait_for_terminal(base_url, print_job_id, timeout_s=args.timeout_s, poll_s=args.poll_s)
    if print_terminal.get("status") != "done":
        raise SystemExit(f"print job failed: {print_terminal}")

    result_obj = print_terminal.get("result")
    if not isinstance(result_obj, dict):
        raise SystemExit(f"unexpected print result: {print_terminal}")
    text = result_obj.get("text")
    if not isinstance(text, str):
        raise SystemExit(f"print result missing text: {print_terminal}")

    print("----- communications.txt (from Redis) -----")
    print(text)
    print("-------------------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
