from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def parse_stage_end(stdout: str, stage_id: str) -> dict | None:
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("event_type") == "stage_end" and obj.get("stage_id") == stage_id:
            return obj
    return None


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def run_once(cli: Path, repo_root: Path, data_dir: Path, env: dict[str, str]) -> dict:
    rc, out, err = run([str(cli), "init", "--path", str(data_dir)], repo_root, env)
    if rc != 0:
        raise RuntimeError(f"init failed: {err or out}")

    embedding_count = 4096
    script = repo_root / "vector_db_v3" / "scripts" / "pipeline_test.py"
    rc, out, err = run(
        [
            sys.executable,
            str(script),
            "--build-dir",
            str(cli.parent),
            "--data-dir",
            str(data_dir),
            "--embedding-count",
            str(embedding_count),
            "--batch-size",
            "256",
            "--input-format",
            "bin",
            "--seed",
            "7",
            "--run-full-pipeline",
        ],
        repo_root / "vector_db_v3",
        env,
    )
    if rc != 0:
        raise RuntimeError(f"pipeline_test failed: {err or out}")

    rc, out_top, err_top = run([str(cli), "build-top-clusters", "--path", str(data_dir), "--seed", "7"], repo_root, env)
    if rc != 0:
        raise RuntimeError(f"build-top-clusters failed: {err_top or out_top}")
    rc, out_mid, err_mid = run([str(cli), "build-mid-layer-clusters", "--path", str(data_dir), "--seed", "7"], repo_root, env)
    if rc != 0:
        raise RuntimeError(f"build-mid-layer-clusters failed: {err_mid or out_mid}")

    top = parse_stage_end(out_top, "top")
    mid = parse_stage_end(out_mid, "mid")
    if top is None or mid is None:
        raise RuntimeError("missing stage_end telemetry for top/mid")

    return {
        "top_elapsed_ms": float(top.get("stage_elapsed_ms", top.get("elapsed_ms", 0.0))),
        "mid_elapsed_ms": float(mid.get("stage_elapsed_ms", mid.get("elapsed_ms", 0.0))),
        "top_alloc_calls": int(top.get("gpu_residency_alloc_calls", 0)),
        "mid_alloc_calls": int(mid.get("gpu_residency_alloc_calls", 0)),
        "top_h2d_saved_est": int(top.get("gpu_residency_bytes_h2d_saved_est", 0)),
        "mid_h2d_saved_est": int(mid.get("gpu_residency_bytes_h2d_saved_est", 0)),
        "top_cache_hits": int(top.get("gpu_residency_cache_hits", 0)),
        "mid_cache_hits": int(mid.get("gpu_residency_cache_hits", 0)),
    }


def summarize(runs: list[dict]) -> dict:
    top = [r["top_elapsed_ms"] for r in runs]
    mid = [r["mid_elapsed_ms"] for r in runs]
    alloc = [r["top_alloc_calls"] + r["mid_alloc_calls"] for r in runs]
    h2d = [r["top_h2d_saved_est"] + r["mid_h2d_saved_est"] for r in runs]
    hits = [r["top_cache_hits"] + r["mid_cache_hits"] for r in runs]
    return {
        "runs": len(runs),
        "top_median_ms": statistics.median(top) if top else 0.0,
        "top_p95_ms": p95(top),
        "mid_median_ms": statistics.median(mid) if mid else 0.0,
        "mid_p95_ms": p95(mid),
        "alloc_calls_median": statistics.median(alloc) if alloc else 0.0,
        "h2d_saved_est_median": statistics.median(h2d) if h2d else 0.0,
        "cache_hits_median": statistics.median(hits) if hits else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Card 3 A/B residency benchmark runner.")
    parser.add_argument("--build-dir", default="vector_db_v3/build")
    parser.add_argument("--out-dir", default="vector_db_v3/gate_evidence/card3_ab")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (repo_root / build_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        raise SystemExit(f"missing cli binary in {build_dir}")

    base_env = dict(os.environ)
    base_env["VECTOR_DB_V3_KMEANS_BACKEND"] = "cuda"
    base_env["VECTOR_DB_V3_KMEANS_PRECISION"] = "auto"
    base_env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"

    modes = {
        "baseline_off": "off",
        "candidate_stage": "stage",
    }

    result = {"runs": args.runs, "warmup_runs": args.warmup_runs, "modes": {}}

    for label, mode in modes.items():
        env = dict(base_env)
        env["VECTOR_DB_V3_GPU_RESIDENCY_MODE"] = mode
        mode_runs: list[dict] = []
        data_root = Path(tempfile.gettempdir()) / f"vectordb_v3_card3_{label}"
        if data_root.exists():
            shutil.rmtree(data_root)
        data_root.mkdir(parents=True, exist_ok=True)

        for idx in range(args.warmup_runs + args.runs):
            run_dir = data_root / f"run_{idx:02d}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            one = run_once(cli, repo_root, run_dir, env)
            if idx >= args.warmup_runs:
                mode_runs.append(one)
        result["modes"][label] = {"mode": mode, "summary": summarize(mode_runs), "samples": mode_runs}

    baseline = result["modes"]["baseline_off"]["summary"]
    candidate = result["modes"]["candidate_stage"]["summary"]
    delta = {
        "top_median_improvement_pct": 0.0,
        "mid_median_improvement_pct": 0.0,
    }
    if baseline["top_median_ms"] > 0:
        delta["top_median_improvement_pct"] = (
            (baseline["top_median_ms"] - candidate["top_median_ms"]) / baseline["top_median_ms"] * 100.0
        )
    if baseline["mid_median_ms"] > 0:
        delta["mid_median_improvement_pct"] = (
            (baseline["mid_median_ms"] - candidate["mid_median_ms"]) / baseline["mid_median_ms"] * 100.0
        )
    result["delta"] = delta

    out_file = out_dir / "card3_residency_ab.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_file}")
    print(json.dumps(result["delta"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
