from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return proc.returncode, proc.stdout, proc.stderr, elapsed_ms
    except FileNotFoundError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return 127, "", str(exc), elapsed_ms


def now_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def tool_version(name: str, arg: str = "--version") -> str:
    try:
        proc = subprocess.run([name, arg], capture_output=True, text=True)
        output = (proc.stdout or proc.stderr).strip()
        return output.splitlines()[0] if output else "unknown"
    except Exception:
        return "unavailable"


def gate_definitions(build_dir: Path, repo_root: Path) -> dict[str, dict]:
    ctest = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
    ]
    return {
        "G1": {
            "mandatory": True,
            "contract_refs": [
                "vector_db_v3/contracts/TEST_GATE_CONTRACT.md",
                "vector_db_v3/contracts/M1_SCOPE_CONTRACT.md",
            ],
            "commands": [
                ctest
                + [
                    "-R",
                    "vectordb_v3_tests|vectordb_v3_exact_search_tests|vectordb_v3_top_layer_artifacts_tests|"
                    "vectordb_v3_mid_layer_artifacts_tests|vectordb_v3_lower_layer_gate_artifacts_tests|"
                    "vectordb_v3_final_layer_artifacts_tests|vectordb_v3_final_layer_eligibility_reconciliation_tests|"
                    "vectordb_v3_kmeans_backend_parity_tests|vectordb_v3_kmeans_tie_break_determinism_tests|"
                    "vectordb_v3_kmeans_backend_selection_tests|vectordb_v3_gpu_residency_tests|"
                    "vectordb_v3_precision_shard_lifecycle_tests|vectordb_v3_precision_shard_alignment_failures_tests|"
                    "vectordb_v3_card5_ingest_async_tests|vectordb_v3_card6_single_process_tests",
                ]
            ],
        },
        "G2": {
            "mandatory": True,
            "contract_refs": [
                "vector_db_v3/contracts/TEST_GATE_CONTRACT.md",
                "vector_db_v3/contracts/M1_SCOPE_CONTRACT.md",
            ],
            "commands": [
                ctest
                + [
                    "-R",
                    "vectordb_v3_durability_wal_tests|vectordb_v3_durability_checkpoint_tests|"
                    "vectordb_v3_durability_replay_crash_tests|vectordb_v3_durability_corruption_tests",
                ]
            ],
        },
        "G3": {
            "mandatory": True,
            "contract_refs": [
                "vector_db_v3/contracts/CLI_CONTRACT.md",
                "vector_db_v3/contracts/ARTIFACT_CONTRACT.md",
                "vector_db_v3/contracts/BINARY_FORMATS.md",
                "vector_db_v3/contracts/TEST_GATE_CONTRACT.md",
            ],
            "commands": [
                ctest
                + [
                    "-R",
                    "vectordb_v3_codec_primitives_tests|vectordb_v3_codec_artifacts_tests|"
                    "vectordb_v3_codec_corruption_tests|vectordb_v3_cli_contract_tests|"
                    "vectordb_v3_top_layer_artifacts_tests|vectordb_v3_mid_layer_artifacts_tests|"
                    "vectordb_v3_lower_layer_gate_artifacts_tests|vectordb_v3_final_layer_artifacts_tests|"
                    "vectordb_v3_final_layer_eligibility_reconciliation_tests|"
                    "vectordb_v3_precision_shard_lifecycle_tests|vectordb_v3_precision_shard_alignment_failures_tests|"
                    "vectordb_v3_card5_ingest_async_tests|vectordb_v3_card6_single_process_tests",
                ]
            ],
        },
        "G4": {
            "mandatory": True,
            "contract_refs": [
                "vector_db_v3/contracts/TEST_GATE_CONTRACT.md",
                "vector_db_v3/contracts/implementationplan.md",
            ],
            "commands": [
                [
                    sys.executable,
                    str(repo_root / "scripts" / "perf_gate.py"),
                    "--build-dir",
                    str(build_dir),
                    "--mode",
                    "baseline",
                    "--profile",
                    "jetson_orin",
                    "--thresholds-file",
                    "__GATE_DIR__/thresholds_jetson_orin.json",
                    "--out-dir",
                    "__GATE_DIR__/baseline",
                ],
                [
                    sys.executable,
                    str(repo_root / "scripts" / "perf_gate.py"),
                    "--build-dir",
                    str(build_dir),
                    "--mode",
                    "hard",
                    "--profile",
                    "jetson_orin",
                    "--thresholds-file",
                    "__GATE_DIR__/thresholds_jetson_orin.json",
                    "--baseline-ref",
                    "__GATE_DIR__/baseline/baseline_metrics.json",
                    "--out-dir",
                    "__GATE_DIR__",
                ]
            ],
        },
        "G5": {
            "mandatory": True,
            "contract_refs": [
                "vector_db_v3/contracts/COMPLIANCE_CONTRACT.md",
                "vector_db_v3/contracts/TRACEABILITY_MATRIX.md",
            ],
            "commands": [
                ctest
                + [
                    "-R",
                    "vectordb_v3_compliance_pass_tests|vectordb_v3_terminal_event_contract_tests|vectordb_v3_cli_contract_tests",
                ]
            ],
        },
        "G6": {
            "mandatory": True,
            "contract_refs": [
                "vector_db_v3/contracts/COMPLIANCE_CONTRACT.md",
                "vector_db_v3/contracts/TEST_GATE_CONTRACT.md",
            ],
            "commands": [
                ctest
                + [
                    "-R",
                    "vectordb_v3_compliance_fail_fast_tests|vectordb_v3_terminal_event_contract_tests|vectordb_v3_cli_contract_tests",
                ]
            ],
        },
        "G7": {
            "mandatory": True,
            "contract_refs": [
                "vector_db_v3/contracts/TERMINAL_EVENT_CONTRACT.md",
                "vector_db_v3/contracts/TEST_GATE_CONTRACT.md",
            ],
            "commands": [
                ctest + ["-R", "vectordb_v3_terminal_event_contract_tests"],
                [
                    sys.executable,
                    str(repo_root / "scripts" / "check_reproducibility.py"),
                    "--build-dir",
                    str(build_dir),
                    "--events-only",
                    "--out-dir",
                    "__GATE_DIR__/repro",
                ],
            ],
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vector_db_v3 G1..G7 gates and emit evidence pack.")
    parser.add_argument("--build-dir", default="vector_db_v3/build")
    parser.add_argument("--profile", choices=["minimal", "full"], default="full")
    parser.add_argument("--fast-fail", action="store_true")
    parser.add_argument("--evidence-root", default="vector_db_v3/gate_evidence")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (repo_root.parent / build_dir).resolve()
    if not build_dir.exists():
        print(f"error: build dir not found: {build_dir}", file=sys.stderr)
        return 2

    run_id = args.run_id or now_run_id()
    evidence_root = Path(args.evidence_root)
    if not evidence_root.is_absolute():
        evidence_root = (repo_root.parent / evidence_root).resolve()
    run_dir = evidence_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_json = {
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cmake": tool_version("cmake"),
        "ctest": tool_version("ctest"),
        "compiler_hint": os.environ.get("CXX", "unknown"),
        "profile": args.profile,
    }
    write_json(run_dir / "environment.json", env_json)

    gates = gate_definitions(build_dir, repo_root)
    active = ["G1", "G3", "G5", "G6", "G7"] if args.profile == "minimal" else ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    gate_map = {
        gate_id: {
            "mandatory": gates[gate_id]["mandatory"],
            "contract_refs": gates[gate_id]["contract_refs"],
            "commands": gates[gate_id]["commands"],
        }
        for gate_id in active
    }
    write_json(run_dir / "gate_map.json", gate_map)

    transcript: list[str] = []
    summary_rows: list[dict] = []
    mandatory_failed = False

    for gate_id in active:
        gate = gates[gate_id]
        gate_dir = run_dir / gate_id
        gate_dir.mkdir(parents=True, exist_ok=True)
        gate_log = gate_dir / "gate.log"
        gate_cmds: list[str] = []
        gate_artifacts = [str(gate_log)]
        gate_start = time.perf_counter()
        gate_failed = False

        with gate_log.open("w", encoding="utf-8") as logf:
            for command in gate["commands"]:
                command = [arg.replace("__GATE_DIR__", str(gate_dir)) for arg in command]
                cmd_txt = " ".join(command)
                gate_cmds.append(cmd_txt)
                transcript.append(f"[{gate_id}] {cmd_txt}")
                logf.write(f"$ {cmd_txt}\n")
                code, out, err, elapsed_ms = run_command(command, cwd=repo_root.parent)
                logf.write(f"exit_code={code} elapsed_ms={elapsed_ms:.3f}\n")
                if out:
                    logf.write(out)
                    if not out.endswith("\n"):
                        logf.write("\n")
                if err:
                    logf.write("--- stderr ---\n")
                    logf.write(err)
                    if not err.endswith("\n"):
                        logf.write("\n")
                if code != 0:
                    gate_failed = True
                    if args.fast_fail:
                        break

        elapsed_gate_ms = (time.perf_counter() - gate_start) * 1000.0
        status = "fail" if gate_failed else "pass"

        row = {
            "gate_id": gate_id,
            "status": status,
            "commands_run": gate_cmds,
            "artifact_paths": gate_artifacts,
            "contract_refs": gate["contract_refs"],
            "duration_ms": round(elapsed_gate_ms, 3),
        }
        summary_rows.append(row)
        write_json(gate_dir / "result.json", row)

        if gate["mandatory"] and gate_failed:
            mandatory_failed = True
            if args.fast_fail:
                break

    transcript_path = run_dir / "commands_transcript.log"
    transcript_path.write_text("\n".join(transcript) + ("\n" if transcript else ""), encoding="utf-8")

    trace_dir = run_dir / "traceability"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "verify_traceability.py"),
        "--matrix",
        str(repo_root / "contracts" / "TRACEABILITY_MATRIX.md"),
        "--gate-map",
        str(run_dir / "gate_map.json"),
        "--out-dir",
        str(trace_dir),
    ]
    t_code, t_out, t_err, _ = run_command(trace_cmd, cwd=repo_root.parent)
    (trace_dir / "verify_traceability.stdout.log").write_text(t_out, encoding="utf-8")
    (trace_dir / "verify_traceability.stderr.log").write_text(t_err, encoding="utf-8")
    trace_ok = t_code == 0
    if not trace_ok:
        mandatory_failed = True

    overall_status = "fail" if mandatory_failed else "pass"
    summary = {
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "overall_status": overall_status,
        "profile": args.profile,
        "traceability_status": "pass" if trace_ok else "fail",
        "gates": summary_rows,
    }
    write_json(run_dir / "summary.json", summary)

    print(json.dumps({"status": overall_status, "run_id": run_id, "evidence_dir": str(run_dir)}, indent=2))
    return 1 if mandatory_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

