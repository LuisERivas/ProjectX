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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        return fail("missing vectordb_v3_cli binary")

    base_dir = Path(tempfile.gettempdir()) / "vectordb_v3_card5_ingest_async"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    rows = [(1000 + i, 0.01 + (i % 9) * 0.001) for i in range(64)]
    bulk_path = base_dir / "bulk.bin"
    write_bulk_bin(bulk_path, rows)

    # Sync baseline ingest.
    sync_dir = base_dir / "sync"
    env_sync = dict(os.environ)
    env_sync["VECTOR_DB_V3_INGEST_ASYNC_MODE"] = "0"
    code, out, err = run([str(cli), "init", "--path", str(sync_dir)], root, env=env_sync)
    if code != 0:
        return fail("sync init failed", out, err)
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(sync_dir), "--input", str(bulk_path), "--batch-size", "8"],
        root,
        env=env_sync,
    )
    if code != 0:
        return fail("sync bulk-insert-bin failed", out, err)
    sync_payload = parse_json(out)
    if sync_payload.get("command") != "bulk-insert-bin" or sync_payload.get("inserted") != len(rows):
        return fail("sync payload mismatch", out, err)

    # Async candidate ingest.
    async_dir = base_dir / "async"
    env_async = dict(os.environ)
    env_async["VECTOR_DB_V3_INGEST_ASYNC_MODE"] = "1"
    env_async["VECTOR_DB_V3_INGEST_PINNED"] = "1"
    code, out, err = run([str(cli), "init", "--path", str(async_dir)], root, env=env_async)
    if code != 0:
        return fail("async init failed", out, err)
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(async_dir), "--input", str(bulk_path), "--batch-size", "8"],
        root,
        env=env_async,
    )
    if code != 0:
        return fail("async bulk-insert-bin failed", out, err)
    async_payload = parse_json(out)
    if async_payload.get("command") != "bulk-insert-bin" or async_payload.get("inserted") != len(rows):
        return fail("async payload mismatch", out, err)
    if async_payload.get("batches") != sync_payload.get("batches"):
        return fail("async/sync batch count mismatch", out, err)

    # Replay consistency after forced mid-run failure.
    fail_dir = base_dir / "forced_fail"
    env_fail = dict(os.environ)
    env_fail["VECTOR_DB_V3_INGEST_ASYNC_MODE"] = "1"
    env_fail["VECTOR_DB_V3_INGEST_FAIL_AFTER_BATCHES"] = "2"
    code, out, err = run([str(cli), "init", "--path", str(fail_dir)], root, env=env_fail)
    if code != 0:
        return fail("forced-fail init failed", out, err)
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(fail_dir), "--input", str(bulk_path), "--batch-size", "8"],
        root,
        env=env_fail,
    )
    if code != 1:
        return fail("forced failure should return exit 1", out, err)
    code, out, err = run([str(cli), "stats", "--path", str(fail_dir)], root, env=dict(os.environ))
    if code != 0:
        return fail("stats after forced failure failed", out, err)
    stats = parse_json(out)
    # 2 batches x 8 rows should be committed before forced failure.
    if stats.get("live_rows") != 16:
        return fail("committed prefix mismatch after forced failure", out, err)

    # Negative malformed binary should remain usage error.
    bad_bin = base_dir / "bad.bin"
    bad_bin.write_bytes(b"\x00\x00\x00\x00")
    code, out, err = run(
        [str(cli), "bulk-insert-bin", "--path", str(sync_dir), "--input", str(bad_bin), "--batch-size", "8"],
        root,
        env=dict(os.environ),
    )
    if code != 2:
        return fail("malformed binary should return exit 2", out, err)
    if not err.startswith("error: "):
        return fail("malformed binary stderr format mismatch", out, err)

    print("vectordb_v3_card5_ingest_async_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
