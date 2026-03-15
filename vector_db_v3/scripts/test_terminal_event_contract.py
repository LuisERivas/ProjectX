from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def fail(msg: str, out: str = "", err: str = "") -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    if out:
        print("--- stdout ---", file=sys.stderr)
        print(out, file=sys.stderr)
    if err:
        print("--- stderr ---", file=sys.stderr)
        print(err, file=sys.stderr)
    return 1


def parse_lines(stdout: str) -> tuple[list[dict], dict | None]:
    events: list[dict] = []
    command_obj: dict | None = None
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, dict) and "event_type" in obj:
            events.append(obj)
        elif isinstance(obj, dict):
            command_obj = obj
    return events, command_obj


def assert_common_fields(events: list[dict]) -> None:
    for e in events:
        for key in [
            "event_type",
            "stage_id",
            "stage_name",
            "status",
            "start_ts",
            "elapsed_ms",
            "pipeline_elapsed_ms",
            "active_pipeline_state",
        ]:
            if key not in e:
                raise AssertionError(f"missing required common field: {key}")
        if e["event_type"] in {"stage_end", "stage_fail", "stage_skip", "pipeline_summary"} and "end_ts" not in e:
            raise AssertionError(f"missing end_ts for terminal event: {e['event_type']}")
        if float(e["elapsed_ms"]) < 0:
            raise AssertionError("elapsed_ms must be non-negative")
        if float(e["pipeline_elapsed_ms"]) < 0:
            raise AssertionError("pipeline_elapsed_ms must be non-negative")


def assert_monotonic_pipeline_elapsed(events: list[dict]) -> None:
    last = 0.0
    for e in events:
        cur = float(e["pipeline_elapsed_ms"])
        if cur < last:
            raise AssertionError("pipeline_elapsed_ms is not monotonic")
        last = cur


def assert_exact_lifecycle(events: list[dict], terminal_event: str) -> None:
    order = [e["event_type"] for e in events]
    expected = ["pipeline_start", "stage_start", terminal_event, "pipeline_summary"]
    if order != expected:
        raise AssertionError(f"unexpected event order: got={order}, expected={expected}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        return fail("missing vectordb_v3_cli binary")

    data_dir = Path(tempfile.gettempdir()) / "vectordb_v3_terminal_event_contract"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    code, out, err = run([str(cli), "init", "--path", str(data_dir)], root)
    if code != 0:
        return fail("init should succeed", out, err)

    # Positive success path (first run baseline unavailable).
    code, out, err = run([str(cli), "build-top-clusters", "--path", str(data_dir)], root)
    if code != 0:
        return fail("build-top-clusters should succeed", out, err)
    events, cmd = parse_lines(out)
    try:
        assert_exact_lifecycle(events, "stage_end")
        assert_common_fields(events)
        assert_monotonic_pipeline_elapsed(events)
        stage_start = next(e for e in events if e["event_type"] == "stage_start")
        if "stage_started_ts" not in stage_start or "stage_elapsed_ms" not in stage_start:
            raise AssertionError("stage_start missing baseline lifecycle fields")
        if stage_start.get("previous_run_available") is not False:
            raise AssertionError("first run should not have previous_run_available=true")
        if stage_start.get("previous_run_stage_elapsed_ms", "missing") is not None:
            raise AssertionError("first run previous_run_stage_elapsed_ms must be null")
    except Exception as exc:
        return fail(f"first run telemetry contract mismatch: {exc}", out, err)
    if not cmd or cmd.get("status") != "ok":
        return fail("final command JSON missing for successful build stage", out, err)

    # Positive success path (second run baseline available).
    code, out, err = run([str(cli), "build-top-clusters", "--path", str(data_dir)], root)
    if code != 0:
        return fail("second build-top-clusters should succeed", out, err)
    events, _ = parse_lines(out)
    try:
        stage_start = next(e for e in events if e["event_type"] == "stage_start")
        if stage_start.get("previous_run_available") is not True:
            raise AssertionError("second run should have previous_run_available=true")
        prev = stage_start.get("previous_run_stage_elapsed_ms")
        if prev is None:
            raise AssertionError("second run previous_run_stage_elapsed_ms must be numeric")
        float(prev)
    except Exception as exc:
        return fail(f"second run baseline contract mismatch: {exc}", out, err)

    # Forced runtime failure path.
    env_fail = dict(os.environ)
    env_fail["VECTOR_DB_V3_FORCE_STAGE_FAIL"] = "1"
    code, out, err = run([str(cli), "build-mid-layer-clusters", "--path", str(data_dir)], root, env=env_fail)
    if code != 1:
        return fail("forced runtime failure should return exit code 1", out, err)
    events, _ = parse_lines(out)
    try:
        assert_exact_lifecycle(events, "stage_fail")
        stage_fail = next(e for e in events if e["event_type"] == "stage_fail")
        if "error_code" not in stage_fail or "error_message" not in stage_fail:
            raise AssertionError("stage_fail missing error fields")
    except Exception as exc:
        return fail(f"runtime stage_fail contract mismatch: {exc}", out, err)

    # Forced compliance failure path.
    env_compliance = dict(os.environ)
    env_compliance["VECTOR_DB_V3_FORCE_COMPLIANCE_FAIL"] = "1"
    code, out, err = run([str(cli), "build-lower-layer-clusters", "--path", str(data_dir)], root, env=env_compliance)
    if code != 1:
        return fail("forced compliance failure should return exit code 1", out, err)
    events, _ = parse_lines(out)
    try:
        stage_fail = next(e for e in events if e["event_type"] == "stage_fail")
        if stage_fail.get("error_code") != "compliance_fail_fast":
            raise AssertionError("compliance stage_fail missing compliance_fail_fast code")
        if "non_compliance_stage" not in stage_fail:
            raise AssertionError("compliance stage_fail missing non_compliance_stage")
    except Exception as exc:
        return fail(f"compliance stage_fail contract mismatch: {exc}", out, err)

    # Forced skip path.
    env_skip = dict(os.environ)
    env_skip["VECTOR_DB_V3_FORCE_STAGE_SKIP"] = "1"
    code, out, err = run([str(cli), "build-final-layer-clusters", "--path", str(data_dir)], root, env=env_skip)
    if code != 0:
        return fail("forced stage skip should still return exit code 0", out, err)
    events, cmd = parse_lines(out)
    try:
        assert_exact_lifecycle(events, "stage_skip")
        summary = next(e for e in events if e["event_type"] == "pipeline_summary")
        if summary.get("final_output_status") != "skipped":
            raise AssertionError("pipeline_summary final_output_status should be skipped")
    except Exception as exc:
        return fail(f"stage_skip contract mismatch: {exc}", out, err)
    if not cmd or cmd.get("status") != "ok":
        return fail("final command JSON missing for skip stage", out, err)

    print("vectordb_v3_terminal_event_contract_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
