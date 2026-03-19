from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def write_bulk_bin(path: Path, rows: list[tuple[int, float]]) -> None:
    dim = 1024
    record_size = 8 + dim * 4
    header = bytearray(18)
    header[0:4] = (0x49423356).to_bytes(4, "little", signed=False)
    header[4:6] = (1).to_bytes(2, "little", signed=False)
    header[6:10] = record_size.to_bytes(4, "little", signed=False)
    header[10:18] = len(rows).to_bytes(8, "little", signed=False)
    with path.open("wb") as f:
        f.write(header)
        for embedding_id, value in rows:
            row = bytearray(record_size)
            row[0:8] = int(embedding_id).to_bytes(8, "little", signed=False)
            packed = struct.pack("<f", float(value))
            for i in range(dim):
                row[8 + i * 4 : 12 + i * 4] = packed
            f.write(row)


def parse_final_json(stdout: str) -> dict:
    command_obj: dict | None = None
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, dict) and "event_type" not in obj:
            command_obj = obj
    if command_obj is None:
        raise ValueError("missing final command json")
    return command_obj


def parse_stage_events(stdout: str) -> list[dict]:
    events: list[dict] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, dict) and "event_type" in obj:
            events.append(obj)
    return events


def fail(message: str, report_path: Path, details: dict) -> int:
    payload = {"status": "fail", "message": message, "details": details}
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Card 6 parity/failure validation script.")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (root / "build").resolve()
    report_path = Path(args.report_out).resolve() if args.report_out else (root / "gate_evidence" / "card6_parity_report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        return fail("missing vectordb_v3_cli binary", report_path, {"build_dir": str(build_dir)})

    root_tmp = Path(tempfile.gettempdir()) / "vectordb_v3_card6_single_process_test"
    if root_tmp.exists():
        shutil.rmtree(root_tmp)
    root_tmp.mkdir(parents=True, exist_ok=True)
    legacy_dir = root_tmp / "legacy_data"
    composite_dir = root_tmp / "composite_data"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    composite_dir.mkdir(parents=True, exist_ok=True)
    bulk_path = root_tmp / "bulk.bin"
    write_bulk_bin(bulk_path, [(90000 + i, 0.0005 * ((i % 41) + 1)) for i in range(384)])

    env_pass = dict(os.environ)
    env_pass["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"

    composite_cmd = [
        str(cli),
        "run-full-pipeline",
        "--path",
        str(composite_dir),
        "--input",
        str(bulk_path),
        "--input-format",
        "bin",
        "--batch-size",
        "64",
        "--seed",
        "7",
    ]
    code, out, err = run(composite_cmd, root.parent, env=env_pass)
    if code != 0:
        return fail("composite command failed", report_path, {"stderr": err, "stdout": out})
    composite_payload = parse_final_json(out)
    if composite_payload.get("stages_completed") != 4:
        return fail("composite stages_completed mismatch", report_path, {"payload": composite_payload})

    legacy_stages = [
        [str(cli), "init", "--path", str(legacy_dir)],
        [str(cli), "bulk-insert-bin", "--path", str(legacy_dir), "--input", str(bulk_path), "--batch-size", "64"],
        [str(cli), "build-top-clusters", "--path", str(legacy_dir), "--seed", "7"],
        [str(cli), "build-mid-layer-clusters", "--path", str(legacy_dir), "--seed", "7"],
        [str(cli), "build-lower-layer-clusters", "--path", str(legacy_dir), "--seed", "7"],
        [str(cli), "build-final-layer-clusters", "--path", str(legacy_dir), "--seed", "7"],
    ]
    legacy_outputs: list[dict] = []
    for cmd in legacy_stages:
        code, out, err = run(cmd, root.parent, env=env_pass)
        legacy_outputs.append({"command": cmd, "exit": code})
        if code != 0:
            return fail("legacy stage command failed", report_path, {"command": cmd, "stderr": err, "stdout": out})

    stats_cmd_legacy = [str(cli), "cluster-stats", "--path", str(legacy_dir)]
    stats_cmd_composite = [str(cli), "cluster-stats", "--path", str(composite_dir)]
    code, out_legacy, err = run(stats_cmd_legacy, root.parent, env=env_pass)
    if code != 0:
        return fail("legacy cluster-stats failed", report_path, {"stderr": err})
    code, out_comp, err = run(stats_cmd_composite, root.parent, env=env_pass)
    if code != 0:
        return fail("composite cluster-stats failed", report_path, {"stderr": err})

    legacy_stats = json.loads(out_legacy)
    composite_stats = json.loads(out_comp)
    if legacy_stats.get("vectors_indexed") != composite_stats.get("vectors_indexed"):
        return fail("vectors_indexed mismatch between legacy and composite", report_path, {"legacy": legacy_stats, "composite": composite_stats})
    if legacy_stats.get("chosen_k") != composite_stats.get("chosen_k"):
        return fail("chosen_k mismatch between legacy and composite", report_path, {"legacy": legacy_stats, "composite": composite_stats})

    force_fail_env = dict(env_pass)
    force_fail_env["VECTOR_DB_V3_FORCE_STAGE_FAIL"] = "1"
    fail_dir = root_tmp / "fail_data"
    fail_cmd = [
        str(cli),
        "run-full-pipeline",
        "--path",
        str(fail_dir),
        "--input",
        str(bulk_path),
        "--input-format",
        "bin",
        "--batch-size",
        "64",
        "--seed",
        "7",
    ]
    code, out, err = run(fail_cmd, root.parent, env=force_fail_env)
    if code != 1:
        return fail("forced stage fail should return exit code 1", report_path, {"exit_code": code, "stderr": err})
    events = parse_stage_events(out)
    if not any(e.get("event_type") == "stage_fail" for e in events):
        return fail("forced stage fail missing stage_fail event", report_path, {"stdout": out})
    if not any(e.get("event_type") == "pipeline_summary" and e.get("status") == "failed" for e in events):
        return fail("forced stage fail missing failed pipeline_summary", report_path, {"stdout": out})

    report = {
        "status": "pass",
        "composite": composite_payload,
        "legacy_stage_count": len(legacy_outputs),
        "legacy_cluster_stats": {
            "vectors_indexed": legacy_stats.get("vectors_indexed"),
            "chosen_k": legacy_stats.get("chosen_k"),
        },
        "composite_cluster_stats": {
            "vectors_indexed": composite_stats.get("vectors_indexed"),
            "chosen_k": composite_stats.get("chosen_k"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("vectordb_v3_card6_single_process_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
