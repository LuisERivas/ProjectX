from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
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


def vec_csv(value: float) -> str:
    return ",".join(f"{value + i * 0.001:.6f}" for i in range(1024))


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

    code, out, err = run([str(cli), "init", "--path", str(data_dir)], root)
    if code != 0:
        return fail("init should succeed", out, err)
    try:
        payload = parse_json(out)
    except Exception as exc:
        return fail(f"init stdout should be json: {exc}", out, err)
    if payload.get("status") != "ok" or payload.get("command") != "init":
        return fail("init json payload mismatch", out, err)

    code, out, err = run([str(cli), "insert", "--path", str(data_dir), "--id", "1", "--vec", vec_csv(1.0)], root)
    if code != 0:
        return fail("insert should succeed", out, err)
    payload = parse_json(out)
    if payload.get("command") != "insert" or payload.get("embedding_id") != 1:
        return fail("insert json payload mismatch", out, err)

    code, out, err = run([str(cli), "get", "--path", str(data_dir), "--id", "1"], root)
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
    )
    if code != 0:
        return fail("bulk-insert should succeed", out, err)
    payload = parse_json(out)
    if payload.get("inserted") != 2:
        return fail("bulk-insert inserted count mismatch", out, err)

    code, out, err = run([str(cli), "search", "--path", str(data_dir), "--vec", vec_csv(1.0), "--topk", "2"], root)
    if code != 0:
        return fail("search should succeed", out, err)
    try:
        arr = parse_json(out)
    except Exception as exc:
        return fail(f"search stdout should be json array: {exc}", out, err)
    if not isinstance(arr, list):
        return fail("search should emit json array", out, err)

    code, out, err = run([str(cli), "delete", "--path", str(data_dir), "--id", "1"], root)
    if code != 0:
        return fail("delete should succeed", out, err)
    payload = parse_json(out)
    if payload.get("command") != "delete" or payload.get("embedding_id") != 1:
        return fail("delete payload mismatch", out, err)

    code, out, err = run([str(cli), "checkpoint", "--path", str(data_dir)], root)
    if code != 0:
        return fail("checkpoint should succeed", out, err)
    payload = parse_json(out)
    if payload.get("command") != "checkpoint":
        return fail("checkpoint payload mismatch", out, err)

    code, out, err = run([str(cli), "insert", "--path", str(data_dir), "--id", "4", "--vec", "1,2,3"], root)
    if code != 2:
        return fail("bad vector should be usage error (2)", out, err)
    if not err.startswith("error: "):
        return fail("bad vector error format", out, err)

    code, out, err = run([str(cli), "unknown-cmd", "--path", str(data_dir)], root)
    if code != 2:
        return fail("unknown command should return 2", out, err)

    print("vectordb_v3_cli_contract_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
