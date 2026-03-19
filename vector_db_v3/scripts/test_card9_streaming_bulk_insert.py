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


def mutate_header_record_count(path: Path, record_count: int) -> None:
    raw = bytearray(path.read_bytes())
    if len(raw) < 18:
        raise ValueError("binary file too short for header mutation")
    raw[10:18] = int(record_count).to_bytes(8, "little", signed=False)
    path.write_bytes(raw)


def truncate_bytes(path: Path, count: int) -> None:
    raw = path.read_bytes()
    if len(raw) <= count:
        raise ValueError("cannot truncate entire file")
    path.write_bytes(raw[:-count])


def append_trailing_bytes(path: Path, payload: bytes) -> None:
    with path.open("ab") as f:
        f.write(payload)


def parse_json(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise ValueError("empty stdout")
    return json.loads(text)


def parse_final_command_json(stdout: str) -> dict:
    payload: dict | None = None
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, dict) and "event_type" not in obj:
            payload = obj
    if payload is None:
        raise ValueError("missing final command payload")
    return payload


def fail(message: str, report_out: Path, functional: dict, negative: dict, streaming: dict) -> int:
    payload = {
        "status": "fail",
        "message": message,
        "functional_results": functional,
        "negative_results": negative,
        "streaming_results": streaming,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Card 9 streaming bulk-insert-bin validation.")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (root / "build").resolve()
    report_out = (
        Path(args.report_out).resolve()
        if args.report_out
        else (root / "gate_evidence" / "card9_streaming_bulk_insert.json").resolve()
    )

    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        return fail("missing vectordb_v3_cli binary", report_out, {}, {}, {})

    functional: dict[str, object] = {"checks": []}
    negative: dict[str, object] = {"checks": []}
    streaming: dict[str, object] = {"checks": []}

    tmp_root = Path(tempfile.gettempdir()) / "vectordb_v3_card9_streaming"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_root / "data"
    bulk_path = tmp_root / "bulk.bin"
    rows = [(95000 + i, 0.001 * ((i % 17) + 1)) for i in range(257)]
    write_bulk_bin(bulk_path, rows)

    env = dict(os.environ)
    env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"

    code, out, err = run([str(cli), "init", "--path", str(data_dir)], root.parent, env=env)
    if code != 0:
        return fail(f"init failed: {err or out}", report_out, functional, negative, streaming)

    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bulk_path), "--batch-size", "64"],
        root.parent,
        env=env,
    )
    if code != 0:
        return fail(f"bulk-insert-bin failed: {err or out}", report_out, functional, negative, streaming)
    payload = parse_json(out)
    functional["checks"].append(
        {
            "name": "bulk_insert_bin_success_payload",
            "pass": bool(
                payload.get("status") == "ok"
                and payload.get("command") == "bulk-insert-bin"
                and int(payload.get("inserted", -1)) == len(rows)
                and int(payload.get("batches", -1)) == 5
            ),
            "payload": payload,
        }
    )
    if not functional["checks"][-1]["pass"]:
        return fail("functional payload check failed", report_out, functional, negative, streaming)

    code, out, err = run(
        [
            str(cli),
            "run-full-pipeline",
            "--path",
            str(tmp_root / "pipeline_data"),
            "--input",
            str(bulk_path),
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
        return fail(f"run-full-pipeline bin failed: {err or out}", report_out, functional, negative, streaming)
    final_payload = parse_final_command_json(out)
    functional["checks"].append(
        {
            "name": "run_full_pipeline_bin",
            "pass": bool(
                final_payload.get("status") == "ok"
                and final_payload.get("command") == "run-full-pipeline"
                and int(final_payload.get("inserted", -1)) == len(rows)
                and int(final_payload.get("stages_completed", -1)) == 4
            ),
            "payload": final_payload,
        }
    )
    if not functional["checks"][-1]["pass"]:
        return fail("run-full-pipeline bin check failed", report_out, functional, negative, streaming)

    bad_payload = tmp_root / "bad_payload.bin"
    write_bulk_bin(bad_payload, [(1, 0.1), (2, 0.2)])
    mutate_header_record_count(bad_payload, 99)
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bad_payload), "--batch-size", "2"],
        root.parent,
        env=env,
    )
    negative["checks"].append({"name": "payload_mismatch", "exit_code": code, "stderr_prefix": err.startswith("error: ")})

    bad_truncated = tmp_root / "bad_truncated.bin"
    write_bulk_bin(bad_truncated, [(3, 0.1), (4, 0.2)])
    truncate_bytes(bad_truncated, 5)
    code2, out2, err2 = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bad_truncated), "--batch-size", "2"],
        root.parent,
        env=env,
    )
    negative["checks"].append({"name": "record_truncated", "exit_code": code2, "stderr_prefix": err2.startswith("error: ")})

    bad_trailing = tmp_root / "bad_trailing.bin"
    write_bulk_bin(bad_trailing, [(5, 0.1), (6, 0.2)])
    append_trailing_bytes(bad_trailing, b"\xAA")
    code3, out3, err3 = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bad_trailing), "--batch-size", "2"],
        root.parent,
        env=env,
    )
    negative["checks"].append({"name": "trailing_bytes", "exit_code": code3, "stderr_prefix": err3.startswith("error: ")})

    code4, out4, err4 = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bulk_path), "--batch-size", "0"],
        root.parent,
        env=env,
    )
    negative["checks"].append({"name": "invalid_batch_size", "exit_code": code4, "stderr_prefix": err4.startswith("error: ")})

    for check in negative["checks"]:
        if check["exit_code"] != 2 or not check["stderr_prefix"]:
            return fail("negative-path classification check failed", report_out, functional, negative, streaming)

    source_text = (root / "cli" / "main.cpp").read_text(encoding="utf-8")
    streaming["checks"].append(
        {
            "name": "no_full_materialization_marker",
            "pass": "std::vector<vector_db_v3::Record> records" not in source_text,
            "details": "bulk-insert-bin should stream batches without full file record vector materialization",
        }
    )
    if not streaming["checks"][-1]["pass"]:
        return fail("streaming proxy check failed", report_out, functional, negative, streaming)

    report = {
        "status": "pass",
        "functional_results": functional,
        "negative_results": negative,
        "streaming_results": streaming,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("vectordb_v3_card9_streaming_bulk_insert_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
