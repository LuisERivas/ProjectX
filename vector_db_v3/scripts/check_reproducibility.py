from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_cli(build_dir: Path) -> Path:
    candidate = build_dir / "vectordb_v3_cli.exe"
    if candidate.exists():
        return candidate
    candidate = build_dir / "vectordb_v3_cli"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("missing vectordb_v3_cli binary")


def parse_final_json(stdout: str) -> dict:
    final: dict | None = None
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict) and "event_type" not in obj:
            final = obj
    return final or {}


def collect_event_fingerprint(stdout: str) -> list[dict]:
    out: list[dict] = []
    # These fields are expected to vary across runs even when behavior is stable.
    volatile = {
        "start_ts",
        "end_ts",
        "elapsed_ms",
        "pipeline_elapsed_ms",
        "stage_elapsed_ms",
        "previous_run_available",
        "baseline_unavailable_reason",
        "previous_run_stage_elapsed_ms",
        "stage_elapsed_delta_ms",
        "stage_started_ts",
        "artifact_path",
    }
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if not isinstance(obj, dict) or "event_type" not in obj:
            continue
        normalized = {k: v for k, v in obj.items() if k not in volatile}
        out.append(normalized)
    return out


def first_event_mismatch(run1_events: list[dict], run2_events: list[dict]) -> dict:
    if len(run1_events) != len(run2_events):
        return {
            "kind": "length_mismatch",
            "run1_events": len(run1_events),
            "run2_events": len(run2_events),
        }

    for idx, (e1, e2) in enumerate(zip(run1_events, run2_events)):
        if e1 == e2:
            continue
        keys = sorted(set(e1.keys()) | set(e2.keys()))
        differing_keys = [k for k in keys if e1.get(k) != e2.get(k)]
        return {
            "kind": "event_mismatch",
            "index": idx,
            "run1_event_type": e1.get("event_type"),
            "run2_event_type": e2.get("event_type"),
            "run1_stage_id": e1.get("stage_id"),
            "run2_stage_id": e2.get("stage_id"),
            "differing_keys": differing_keys[:32],
            "run1_event": e1,
            "run2_event": e2,
        }
    return {"kind": "none"}


def write_seeded_jsonl(path: Path, count: int = 64, dim: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for embedding_id in range(1, count + 1):
            vec = [round(((embedding_id * 31 + j * 17) % 1000) / 1000.0, 6) for j in range(dim)]
            f.write(json.dumps({"embedding_id": embedding_id, "vector": vec}) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def collect_bin_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*.bin")):
        rel = path.relative_to(root).as_posix()
        hashes[rel] = sha256_file(path)
    return hashes


def run_pipeline(cli: Path, repo_root: Path, data_dir: Path, events_only: bool, env: dict[str, str]) -> tuple[bool, dict]:
    if data_dir.exists():
        shutil.rmtree(data_dir)
    bulk_jsonl = data_dir / "bulk.jsonl"
    write_seeded_jsonl(bulk_jsonl)

    steps = [
        [str(cli), "init", "--path", str(data_dir)],
        [str(cli), "bulk-insert", "--path", str(data_dir), "--input", str(bulk_jsonl), "--batch-size", "16"],
        [str(cli), "build-top-clusters", "--path", str(data_dir), "--seed", "7"],
        [str(cli), "build-mid-layer-clusters", "--path", str(data_dir), "--seed", "7"],
        [str(cli), "build-lower-layer-clusters", "--path", str(data_dir), "--seed", "7"],
        [str(cli), "build-final-layer-clusters", "--path", str(data_dir), "--seed", "7"],
    ]
    event_fingerprints: list[dict] = []
    command_results: list[dict] = []
    for step in steps:
        code, out, err = run(step, cwd=repo_root, env=env)
        command_results.append({"command": " ".join(step), "exit_code": code, "stderr": err})
        if code != 0:
            return False, {"commands": command_results}
        event_fingerprints.extend(collect_event_fingerprint(out))
        final_payload = parse_final_json(out)
        if final_payload.get("status") not in {"ok", None}:
            return False, {"commands": command_results}

    result = {"commands": command_results, "event_fingerprints": event_fingerprints}
    if not events_only:
        result["bin_hashes"] = collect_bin_hashes(data_dir)
    return True, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatability checks for deterministic v3 paths.")
    parser.add_argument("--build-dir", default="vector_db_v3/build")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--events-only", action="store_true")
    args = parser.parse_args()

    module_root = Path(__file__).resolve().parents[1]
    repo_root = module_root.parent

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (repo_root / build_dir).resolve()
    if not build_dir.exists():
        print(f"error: build dir not found: {build_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else (module_root / "gate_evidence" / "repro_ad_hoc")
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cli = resolve_cli(build_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    base = Path(tempfile.gettempdir()) / "vectordb_v3_repro_check"
    run1_dir = base / "run1"
    run2_dir = base / "run2"
    env = dict(os.environ)
    env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"

    ok1, one = run_pipeline(cli, repo_root, run1_dir, args.events_only, env)
    ok2, two = run_pipeline(cli, repo_root, run2_dir, args.events_only, env)
    if not ok1 or not ok2:
        diff = {"status": "fail", "reason": "pipeline command failed", "run1": one, "run2": two}
        write_json(out_dir / "diff_report.json", diff)
        write_json(out_dir / "result.json", {"status": "fail", "events_only": args.events_only})
        print(json.dumps({"status": "fail"}, indent=2))
        return 1

    events_match = one["event_fingerprints"] == two["event_fingerprints"]
    bins_match = True
    if not args.events_only:
        bins_match = one.get("bin_hashes", {}) == two.get("bin_hashes", {})

    status = "pass" if events_match and bins_match else "fail"
    first_mismatch = first_event_mismatch(one["event_fingerprints"], two["event_fingerprints"])
    diff_report = {
        "status": status,
        "events_match": events_match,
        "bins_match": bins_match,
        "run1_events": len(one["event_fingerprints"]),
        "run2_events": len(two["event_fingerprints"]),
        "run1_hash_count": len(one.get("bin_hashes", {})),
        "run2_hash_count": len(two.get("bin_hashes", {})),
        "first_event_mismatch": first_mismatch,
    }
    write_json(out_dir / "diff_report.json", diff_report)
    write_json(
        out_dir / "result.json",
        {
            "status": status,
            "events_only": args.events_only,
            "artifact_paths": [str(out_dir / "result.json"), str(out_dir / "diff_report.json")],
        },
    )
    print(json.dumps(diff_report, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

