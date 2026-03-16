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
    lifecycle = {"pipeline_start", "stage_start", "stage_end", "stage_fail", "stage_skip", "pipeline_summary"}
    order = [e["event_type"] for e in events if e["event_type"] in lifecycle]
    expected = ["pipeline_start", "stage_start", terminal_event, "pipeline_summary"]
    if order != expected:
        raise AssertionError(f"unexpected event order: got={order}, expected={expected}")


def assert_top_stage_sub_events(events: list[dict]) -> None:
    k_selection = [e for e in events if e.get("event_type") == "k_selection"]
    if not k_selection:
        raise AssertionError("missing k_selection event")
    latest = k_selection[-1]
    for key in ["k_min", "k_max", "chosen_k", "tested_ks"]:
        if key not in latest:
            raise AssertionError(f"k_selection missing {key}")
    if int(latest["k_min"]) <= 0 or int(latest["k_max"]) <= 0:
        raise AssertionError("k bounds must be positive")
    if int(latest["chosen_k"]) < int(latest["k_min"]) or int(latest["chosen_k"]) > int(latest["k_max"]):
        raise AssertionError("chosen_k must be within [k_min,k_max]")

    artifact_writes = [e for e in events if e.get("event_type") == "artifact_write"]
    expected = {
        "id_estimate.bin",
        "elbow_trace.bin",
        "centroids.bin",
        "assignments.bin",
        "stability_report.bin",
        "cluster_manifest.bin",
    }
    observed = set()
    for e in artifact_writes:
        path = str(e.get("artifact_path", ""))
        for name in expected:
            if path.endswith(name):
                observed.add(name)
    if observed != expected:
        raise AssertionError(f"artifact_write coverage mismatch: observed={sorted(observed)}")


def assert_mid_stage_sub_events(events: list[dict]) -> None:
    artifact_writes = [e for e in events if e.get("event_type") == "artifact_write"]
    observed = set()
    for e in artifact_writes:
        path = str(e.get("artifact_path", ""))
        if path.endswith("mid_layer_clustering/assignments.bin"):
            observed.add("mid_assignments")
        if path.endswith("mid_layer_clustering/MID_LAYER_CLUSTERING.bin"):
            observed.add("mid_summary")
    if observed != {"mid_assignments", "mid_summary"}:
        raise AssertionError(f"mid artifact_write coverage mismatch: observed={sorted(observed)}")

    progress_events = [e for e in events if e.get("event_type") == "stage_progress"]
    for event in progress_events:
        if "centroid_id" not in event or "job_id" not in event:
            raise AssertionError("mid stage_progress events must include centroid_id and job_id")


def assert_lower_stage_sub_events(events: list[dict]) -> None:
    artifact_writes = [e for e in events if e.get("event_type") == "artifact_write"]
    observed = False
    for e in artifact_writes:
        path = str(e.get("artifact_path", ""))
        if path.endswith("lower_layer_clustering/LOWER_LAYER_CLUSTERING.bin"):
            observed = True
            break
    if not observed:
        raise AssertionError("lower artifact_write coverage mismatch")

    progress_events = [e for e in events if e.get("event_type") == "stage_progress"]
    if not progress_events:
        raise AssertionError("lower stage_progress events missing")
    for event in progress_events:
        if "centroid_id" not in event or "job_id" not in event:
            raise AssertionError("lower stage_progress events must include centroid_id and job_id")
        if event.get("job_phase") == "end" and "gate_decision" not in event:
            raise AssertionError("lower stage end progress must include gate_decision")


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
        assert_top_stage_sub_events(events)
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

    # Mid-stage success path.
    code, out, err = run([str(cli), "build-mid-layer-clusters", "--path", str(data_dir)], root)
    if code != 0:
        return fail("build-mid-layer-clusters should succeed", out, err)
    events, cmd = parse_lines(out)
    try:
        assert_exact_lifecycle(events, "stage_end")
        assert_common_fields(events)
        assert_monotonic_pipeline_elapsed(events)
        assert_mid_stage_sub_events(events)
    except Exception as exc:
        return fail(f"mid run telemetry contract mismatch: {exc}", out, err)
    if not cmd or cmd.get("status") != "ok":
        return fail("final command JSON missing for successful mid build stage", out, err)

    # Lower-stage success path.
    code, out, err = run([str(cli), "build-lower-layer-clusters", "--path", str(data_dir)], root)
    if code != 0:
        return fail("build-lower-layer-clusters should succeed", out, err)
    events, cmd = parse_lines(out)
    try:
        assert_exact_lifecycle(events, "stage_end")
        assert_common_fields(events)
        assert_monotonic_pipeline_elapsed(events)
        assert_lower_stage_sub_events(events)
    except Exception as exc:
        return fail(f"lower run telemetry contract mismatch: {exc}", out, err)
    if not cmd or cmd.get("status") != "ok":
        return fail("final command JSON missing for successful lower build stage", out, err)

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
