from __future__ import annotations

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


def fail(msg: str, out: str = "", err: str = "") -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    if out:
        print("--- stdout ---", file=sys.stderr)
        print(out, file=sys.stderr)
    if err:
        print("--- stderr ---", file=sys.stderr)
        print(err, file=sys.stderr)
    return 1


def parse_json(stdout: str):
    text = stdout.strip()
    if not text:
        raise ValueError("empty stdout")
    return json.loads(text)


def parse_final_command_json(stdout: str):
    command_obj = None
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


def vec_csv(value: float) -> str:
    return ",".join(f"{value + i * 0.001:.6f}" for i in range(1024))


def vec_const_csv(value: float) -> str:
    return ",".join(f"{value:.6f}" for _ in range(1024))


def write_bulk_bin(path: Path, rows: list[tuple[int, float]]) -> None:
    # Header: magic(u32), version(u16), record_size(u32), record_count(u64)
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
    if count <= 0:
        return
    raw = path.read_bytes()
    if len(raw) <= count:
        raise ValueError("cannot truncate entire file")
    path.write_bytes(raw[:-count])


def append_trailing_bytes(path: Path, payload: bytes) -> None:
    with path.open("ab") as f:
        f.write(payload)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        return fail("missing vectordb_v3_cli binary")

    data_dir = Path(tempfile.gettempdir()) / "vectordb_v3_cli_contract"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    env_pass = dict(os.environ)
    env_pass["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"

    code, out, err = run([str(cli), "init", "--path", str(data_dir)], root, env=env_pass)
    if code != 0:
        return fail("init should succeed", out, err)
    try:
        payload = parse_json(out)
    except Exception as exc:
        return fail(f"init stdout should be json: {exc}", out, err)
    if payload.get("status") != "ok" or payload.get("command") != "init":
        return fail("init json payload mismatch", out, err)

    code, out, err = run([str(cli), "insert", "--path", str(data_dir), "--id", "1", "--vec", vec_csv(1.0)], root, env=env_pass)
    if code != 0:
        return fail("insert should succeed", out, err)
    payload = parse_json(out)
    if payload.get("command") != "insert" or payload.get("embedding_id") != 1:
        return fail("insert json payload mismatch", out, err)

    code, out, err = run([str(cli), "get", "--path", str(data_dir), "--id", "1"], root, env=env_pass)
    if code != 0:
        return fail("get should succeed", out, err)
    payload = parse_json(out)
    if payload.get("embedding_id") != 1 or len(payload.get("vector", [])) != 1024:
        return fail("get payload mismatch", out, err)

    jsonl = data_dir / "bulk.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"embedding_id": 2, "vector": [0.25] * 1024}) + "\n")
        f.write(json.dumps({"embedding_id": 3, "vector": [0.5] * 1024}) + "\n")
    code, out, err = run(
        [str(cli), "bulk-insert", "--path", str(data_dir), "--input", str(jsonl), "--batch-size", "1"],
        root,
        env=env_pass,
    )
    if code != 0:
        return fail("bulk-insert should succeed", out, err)
    payload = parse_json(out)
    if payload.get("inserted") != 2:
        return fail("bulk-insert inserted count mismatch", out, err)

    bulk_bin = data_dir / "bulk.bin"
    write_bulk_bin(bulk_bin, [(20, 0.2), (21, 0.3)])
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bulk_bin), "--batch-size", "1"],
        root,
        env=env_pass,
    )
    if code != 0:
        return fail("bulk-insert-bin should succeed", out, err)
    payload = parse_json(out)
    if payload.get("command") != "bulk-insert-bin" or payload.get("inserted") != 2:
        return fail("bulk-insert-bin payload mismatch", out, err)

    pipeline_dir = Path(tempfile.gettempdir()) / "vectordb_v3_cli_contract_pipeline"
    if pipeline_dir.exists():
        shutil.rmtree(pipeline_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_bin = pipeline_dir / "bulk_pipeline.bin"
    write_bulk_bin(pipeline_bin, [(1000 + i, 0.0001 * ((i % 31) + 1)) for i in range(256)])
    code, out, err = run(
        [
            str(cli),
            "run-full-pipeline",
            "--path",
            str(pipeline_dir),
            "--input",
            str(pipeline_bin),
            "--input-format",
            "bin",
            "--batch-size",
            "64",
            "--seed",
            "7",
        ],
        root,
        env=env_pass,
    )
    if code != 0:
        return fail("run-full-pipeline should succeed", out, err)
    payload = parse_final_command_json(out)
    if payload.get("command") != "run-full-pipeline" or payload.get("status") != "ok":
        return fail("run-full-pipeline payload mismatch", out, err)
    if payload.get("inserted") != 256 or payload.get("stages_completed") != 4:
        return fail("run-full-pipeline inserted/stage counts mismatch", out, err)

    code, out, err = run(
        [str(cli), "run-full-pipeline", "--path", str(pipeline_dir), "--input", str(pipeline_bin)],
        root,
        env=env_pass,
    )
    if code != 2:
        return fail("run-full-pipeline missing --input-format should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("run-full-pipeline usage stderr format mismatch", out, err)

    bad_bin = data_dir / "bulk_bad.bin"
    bad_bin.write_bytes(b"\x00\x00\x00\x00")
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bad_bin), "--batch-size", "1"],
        root,
        env=env_pass,
    )
    if code != 2:
        return fail("bulk-insert-bin malformed header should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("bulk-insert-bin malformed header stderr format", out, err)

    bad_payload_bin = data_dir / "bulk_bad_payload.bin"
    write_bulk_bin(bad_payload_bin, [(30, 0.2), (31, 0.3)])
    mutate_header_record_count(bad_payload_bin, 99)
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bad_payload_bin), "--batch-size", "1"],
        root,
        env=env_pass,
    )
    if code != 2:
        return fail("bulk-insert-bin payload mismatch should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("bulk-insert-bin payload mismatch stderr format", out, err)

    bad_truncated_bin = data_dir / "bulk_bad_truncated.bin"
    write_bulk_bin(bad_truncated_bin, [(40, 0.2), (41, 0.3)])
    truncate_bytes(bad_truncated_bin, 7)
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bad_truncated_bin), "--batch-size", "2"],
        root,
        env=env_pass,
    )
    if code != 2:
        return fail("bulk-insert-bin truncated record should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("bulk-insert-bin truncated record stderr format", out, err)

    bad_trailing_bin = data_dir / "bulk_bad_trailing.bin"
    write_bulk_bin(bad_trailing_bin, [(50, 0.2), (51, 0.3)])
    append_trailing_bytes(bad_trailing_bin, b"\xFF")
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bad_trailing_bin), "--batch-size", "2"],
        root,
        env=env_pass,
    )
    if code != 2:
        return fail("bulk-insert-bin trailing-bytes should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("bulk-insert-bin trailing-bytes stderr format", out, err)

    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bulk_bin), "--batch-size", "0"],
        root,
        env=env_pass,
    )
    if code != 2:
        return fail("bulk-insert-bin batch-size 0 should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("bulk-insert-bin batch-size 0 stderr format", out, err)

    code, out, err = run([str(cli), "search", "--path", str(data_dir), "--vec", vec_csv(1.0), "--topk", "2"], root, env=env_pass)
    if code != 0:
        return fail("search should succeed", out, err)
    try:
        arr = parse_json(out)
    except Exception as exc:
        return fail(f"search stdout should be json array: {exc}", out, err)
    if not isinstance(arr, list):
        return fail("search should emit json array", out, err)
    # Scores must be non-increasing for exact search ranking.
    for i in range(1, len(arr)):
        if arr[i - 1]["score"] < arr[i]["score"]:
            return fail("search scores are not sorted descending", out, err)

    # Insert tie-case records and verify deterministic tie-break by embedding_id ascending.
    tie_vec = vec_const_csv(1.0)
    code, out, err = run([str(cli), "insert", "--path", str(data_dir), "--id", "100", "--vec", tie_vec], root, env=env_pass)
    if code != 0:
        return fail("insert tie record 100 should succeed", out, err)
    code, out, err = run([str(cli), "insert", "--path", str(data_dir), "--id", "101", "--vec", tie_vec], root, env=env_pass)
    if code != 0:
        return fail("insert tie record 101 should succeed", out, err)
    zero_query = vec_const_csv(0.0)
    code, out, err = run([str(cli), "search", "--path", str(data_dir), "--vec", zero_query, "--topk", "5"], root, env=env_pass)
    if code != 0:
        return fail("zero-query search should succeed", out, err)
    arr = parse_json(out)
    if len(arr) < 5:
        return fail("zero-query search should return at least 5 rows", out, err)
    # All scores are ties at zero -> must sort by embedding_id ascending.
    ids = [row["embedding_id"] for row in arr]
    if ids != sorted(ids):
        return fail("tie-break ordering should be embedding_id ascending", out, err)

    code, out, err = run([str(cli), "delete", "--path", str(data_dir), "--id", "1"], root, env=env_pass)
    if code != 0:
        return fail("delete should succeed", out, err)
    payload = parse_json(out)
    if payload.get("command") != "delete" or payload.get("embedding_id") != 1:
        return fail("delete payload mismatch", out, err)

    code, out, err = run([str(cli), "checkpoint", "--path", str(data_dir)], root, env=env_pass)
    if code != 0:
        return fail("checkpoint should succeed", out, err)
    payload = parse_json(out)
    if payload.get("command") != "checkpoint":
        return fail("checkpoint payload mismatch", out, err)

    code, out, err = run([str(cli), "build-top-clusters", "--path", str(data_dir), "--seed", "7"], root, env=env_pass)
    if code != 0:
        return fail("build-top-clusters should succeed", out, err)
    payload = parse_final_command_json(out)
    if payload.get("status") != "ok" or payload.get("command") != "build-top":
        return fail("build-top-clusters command payload mismatch", out, err)

    code, out, err = run([str(cli), "build-mid-layer-clusters", "--path", str(data_dir), "--seed", "7"], root, env=env_pass)
    if code != 0:
        return fail("build-mid-layer-clusters should succeed", out, err)
    payload = parse_final_command_json(out)
    if payload.get("status") != "ok" or payload.get("command") != "build-mid":
        return fail("build-mid-layer-clusters command payload mismatch", out, err)

    code, out, err = run([str(cli), "build-lower-layer-clusters", "--path", str(data_dir), "--seed", "7"], root, env=env_pass)
    if code != 0:
        return fail("build-lower-layer-clusters should succeed", out, err)
    payload = parse_final_command_json(out)
    if payload.get("status") != "ok" or payload.get("command") != "build-lower":
        return fail("build-lower-layer-clusters command payload mismatch", out, err)

    code, out, err = run([str(cli), "build-final-layer-clusters", "--path", str(data_dir), "--seed", "7"], root, env=env_pass)
    if code != 0:
        return fail("build-final-layer-clusters should succeed", out, err)
    payload = parse_final_command_json(out)
    if payload.get("status") != "ok" or payload.get("command") != "build-final":
        return fail("build-final-layer-clusters command payload mismatch", out, err)

    code, out, err = run([str(cli), "cluster-stats", "--path", str(data_dir)], root, env=env_pass)
    if code != 0:
        return fail("cluster-stats should succeed", out, err)
    payload = parse_json(out)
    required = [
        "cuda_required",
        "cuda_enabled",
        "tensor_core_required",
        "tensor_core_active",
        "gpu_arch_class",
        "kernel_backend_path",
        "hot_path_language",
        "compliance_status",
        "fallback_reason",
        "non_compliance_stage",
    ]
    for key in required:
        if key not in payload:
            return fail(f"cluster-stats missing compliance field: {key}", out, err)
    if payload.get("compliance_status") != "pass":
        return fail("cluster-stats compliance_status should be pass under pass profile", out, err)

    env_fail = dict(os.environ)
    env_fail["VECTOR_DB_V3_FORCE_COMPLIANCE_FAIL"] = "1"
    code, out, err = run([str(cli), "build-top-clusters", "--path", str(data_dir), "--seed", "7"], root, env=env_fail)
    if code != 1:
        return fail("forced compliance failure should return 1", out, err)
    if not err.startswith("error: "):
        return fail("forced compliance failure stderr format", out, err)

    code, out, err = run([str(cli), "insert", "--path", str(data_dir), "--id", "4", "--vec", "1,2,3"], root, env=env_pass)
    if code != 2:
        return fail("bad vector should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("bad vector error format", out, err)

    code, out, err = run([str(cli), "unknown-cmd", "--path", str(data_dir)], root, env=env_pass)
    if code != 2:
        return fail("unknown command should return 2", out, err)

    print("vectordb_v3_cli_contract_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
