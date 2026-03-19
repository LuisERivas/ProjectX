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


def parse_json(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise ValueError("empty stdout")
    return json.loads(text)


def parse_final_json(stdout: str) -> dict:
    payload: dict | None = None
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if isinstance(obj, dict) and "event_type" not in obj:
            payload = obj
    if payload is None:
        raise ValueError("missing final command payload")
    return payload


def fail(message: str, details: dict, report_out: Path) -> int:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps({"status": "fail", "message": message, "details": details}, indent=2) + "\n", encoding="utf-8")
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Card 7 WAL commit policy matrix.")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (root / "build").resolve()
    report_out = Path(args.report_out).resolve() if args.report_out else (root / "gate_evidence" / "card7_policy_matrix.json").resolve()

    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        return fail("missing vectordb_v3_cli binary", {"build_dir": str(build_dir)}, report_out)

    tmp_root = Path(tempfile.gettempdir()) / "vectordb_v3_card7_policy_matrix"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    input_path = tmp_root / "bulk.bin"
    rows = [(80000 + i, 0.0001 * ((i % 59) + 1)) for i in range(320)]
    write_bulk_bin(input_path, rows)

    mode_reports: list[dict] = []
    for mode in ("strict_per_record", "batch_group_commit"):
        env = dict(os.environ)
        env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
        env["VECTOR_DB_V3_WAL_COMMIT_POLICY"] = mode

        data_dir = tmp_root / f"data_{mode}"
        code, out, err = run([str(cli), "init", "--path", str(data_dir)], root.parent, env=env)
        if code != 0:
            return fail("init failed", {"mode": mode, "stderr": err, "stdout": out}, report_out)

        code, out, err = run(
            [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(input_path), "--batch-size", "64"],
            root.parent,
            env=env,
        )
        if code != 0:
            return fail("bulk-insert-bin failed", {"mode": mode, "stderr": err, "stdout": out}, report_out)
        bulk_payload = parse_json(out)
        if int(bulk_payload.get("inserted", -1)) != len(rows):
            return fail("inserted count mismatch", {"mode": mode, "payload": bulk_payload}, report_out)

        code, out, err = run([str(cli), "stats", "--path", str(data_dir)], root.parent, env=env)
        if code != 0:
            return fail("stats failed", {"mode": mode, "stderr": err, "stdout": out}, report_out)
        stats_payload = parse_json(out)
        if int(stats_payload.get("live_rows", -1)) != len(rows):
            return fail("live_rows mismatch", {"mode": mode, "payload": stats_payload}, report_out)

        # Card 6 non-regression: composite command remains valid with policy mode.
        composite_dir = tmp_root / f"composite_{mode}"
        code, out, err = run(
            [
                str(cli),
                "run-full-pipeline",
                "--path",
                str(composite_dir),
                "--input",
                str(input_path),
                "--input-format",
                "bin",
                "--batch-size",
                "64",
                "--seed",
                "7",
            ],
            root.parent,
            env=env,
        )
        if code != 0:
            return fail("run-full-pipeline failed under policy mode", {"mode": mode, "stderr": err, "stdout": out}, report_out)
        pipeline_payload = parse_final_json(out)
        if pipeline_payload.get("status") != "ok" or pipeline_payload.get("command") != "run-full-pipeline":
            return fail("composite payload mismatch", {"mode": mode, "payload": pipeline_payload}, report_out)

        mode_reports.append(
            {
                "mode": mode,
                "bulk_inserted": int(bulk_payload.get("inserted", 0)),
                "bulk_batches": int(bulk_payload.get("batches", 0)),
                "live_rows": int(stats_payload.get("live_rows", 0)),
                "composite_stages_completed": int(pipeline_payload.get("stages_completed", 0)),
            }
        )

    report = {"status": "pass", "modes": mode_reports}
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("vectordb_v3_card7_wal_commit_policy_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
