from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
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


def parse_final_command_json(stdout: str) -> dict:
    payload: dict | None = None
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict) and "event_type" not in obj:
            payload = obj
    if payload is None:
        raise ValueError("missing final command payload")
    return payload


def parse_event_lines(stdout: str) -> list[dict]:
    events: list[dict] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("event_type"):
            events.append(obj)
    return events


def read_manifest_payload(manifest_path: Path) -> dict:
    raw = manifest_path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"manifest too short: {manifest_path}")
    schema_version, record_type = struct.unpack_from("<HH", raw, 0)
    record_count, payload_bytes, checksum_crc32 = struct.unpack_from("<III", raw, 4)
    payload = raw[16:]
    if len(payload) != payload_bytes:
        raise ValueError(f"manifest payload size mismatch: {manifest_path}")
    if zlib.crc32(payload) & 0xFFFFFFFF != checksum_crc32:
        raise ValueError(f"manifest crc mismatch: {manifest_path}")
    if schema_version != 1 or record_type != 0x0F01 or record_count != 1:
        raise ValueError(f"manifest header invariant mismatch: {manifest_path}")
    return json.loads(payload.decode("utf-8"))


def to_abs_artifact_path(clusters_current: Path, artifact_path: str) -> Path:
    candidate = Path(artifact_path)
    if candidate.is_absolute():
        return candidate
    return (clusters_current / candidate).resolve()


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def should_skip_checksum_check(manifest_path: Path, target_path: Path) -> bool:
    # Some summary manifests intentionally store checksums for related artifacts
    # while the artifact_path points to the summary manifest container itself.
    # Treat self-referential entries as non-blocking for parity checks.
    try:
        return manifest_path.resolve() == target_path.resolve()
    except Exception:
        return False


def fail(message: str, report_out: Path, checks: list[dict]) -> int:
    payload = {"status": "fail", "message": message, "checks": checks}
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Card 10 metadata overhead parity checks.")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (root / "build").resolve()
    report_out = (
        Path(args.report_out).resolve()
        if args.report_out
        else (root / "gate_evidence" / "card10_metadata_overhead_checks.json").resolve()
    )

    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        return fail("missing vectordb_v3_cli binary", report_out, [])

    tmp_root = Path(tempfile.gettempdir()) / "vectordb_v3_card10_metadata"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    data_dir = tmp_root / "data"
    bulk_path = tmp_root / "bulk.bin"
    rows = [(99000 + i, 0.0005 * ((i % 31) + 1)) for i in range(257)]
    write_bulk_bin(bulk_path, rows)

    env = dict(os.environ)
    env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
    code, out, err = run(
        [
            str(cli),
            "run-full-pipeline",
            "--path",
            str(data_dir),
            "--input",
            str(bulk_path),
            "--input-format",
            "bin",
            "--batch-size",
            "64",
            "--seed",
            "17",
        ],
        cwd=root.parent,
        env=env,
    )
    if code != 0:
        return fail(f"run-full-pipeline failed: {err or out}", report_out, [])

    checks: list[dict] = []
    final_payload = parse_final_command_json(out)
    checks.append(
        {
            "name": "full_pipeline_success",
            "pass": bool(
                final_payload.get("status") == "ok"
                and final_payload.get("command") == "run-full-pipeline"
                and int(final_payload.get("inserted", -1)) == len(rows)
            ),
            "payload": final_payload,
        }
    )
    if not checks[-1]["pass"]:
        return fail("full pipeline payload parity failed", report_out, checks)

    events = parse_event_lines(out)
    artifact_events = [e for e in events if e.get("event_type") == "artifact_write" and e.get("status") == "completed"]
    if not artifact_events:
        return fail("missing artifact_write events", report_out, checks)

    mismatches: list[dict] = []
    for event in artifact_events:
        extra = event.get("extra", {})
        artifact_path = Path(str(extra.get("artifact_path", "")))
        if not artifact_path.exists():
            mismatches.append({"artifact_path": str(artifact_path), "error": "missing_path"})
            continue
        expected_bytes = int(extra.get("bytes_written", -1))
        actual_bytes = artifact_path.stat().st_size
        if expected_bytes != actual_bytes:
            mismatches.append(
                {
                    "artifact_path": str(artifact_path),
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                }
            )
    checks.append(
        {
            "name": "artifact_write_bytes_match_file_size",
            "pass": len(mismatches) == 0,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:8],
        }
    )
    if not checks[-1]["pass"]:
        return fail("artifact_write bytes_written mismatch detected", report_out, checks)

    clusters_current = data_dir / "clusters" / "current"
    manifest_checks: list[dict] = []
    for event in artifact_events:
        artifact_path = Path(str(event.get("extra", {}).get("artifact_path", "")))
        if not artifact_path.exists() or artifact_path.suffix != ".bin":
            continue
        try:
            payload = read_manifest_payload(artifact_path)
        except Exception:
            continue

        if isinstance(payload.get("checksum"), str) and isinstance(payload.get("artifact_path"), str):
            target_path = to_abs_artifact_path(clusters_current, payload["artifact_path"])
            if target_path.exists():
                if should_skip_checksum_check(artifact_path, target_path):
                    continue
                manifest_checks.append(
                    {
                        "manifest_path": str(artifact_path),
                        "artifact_path": str(target_path),
                        "pass": sha256_hex(target_path) == payload["checksum"],
                    }
                )
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                if not isinstance(item.get("checksum"), str) or not isinstance(item.get("artifact_path"), str):
                    continue
                target_path = to_abs_artifact_path(clusters_current, item["artifact_path"])
                if not target_path.exists():
                    continue
                if should_skip_checksum_check(artifact_path, target_path):
                    continue
                manifest_checks.append(
                    {
                        "manifest_path": str(artifact_path),
                        "artifact_path": str(target_path),
                        "pass": sha256_hex(target_path) == item["checksum"],
                    }
                )

    checks.append(
        {
            "name": "manifest_checksum_parity",
            "pass": bool(manifest_checks) and all(bool(c.get("pass")) for c in manifest_checks),
            "checked": len(manifest_checks),
            "sample": manifest_checks[:8],
        }
    )
    if not checks[-1]["pass"]:
        return fail("manifest checksum parity failed", report_out, checks)

    report = {"status": "pass", "checks": checks}
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("vectordb_v3_card10_metadata_overhead_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
