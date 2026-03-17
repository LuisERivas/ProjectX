from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
import random
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

VECTOR_DIM = 1024
BIN_MAGIC_V3BI = 0x49423356
BIN_VERSION = 1
BIN_HEADER_BYTES = 18  # u32 magic + u16 version + u32 record_size + u64 record_count
BIN_RECORD_BYTES = 8 + VECTOR_DIM * 4  # u64 embedding_id + 1024 * f32
MANIFEST_HEADER_BYTES = 16


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_ms: float


def emit_usage_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def emit_runtime_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def log_stage_start(stage_name: str) -> None:
    print(f"[START] {stage_name}")


def log_stage_end(stage_name: str, result: CommandResult) -> None:
    if result.exit_code == 0:
        print(f"[OK] {stage_name} | latency_ms={result.elapsed_ms:.3f}")
    else:
        print(f"[FAIL] {stage_name} | latency_ms={result.elapsed_ms:.3f} | exit={result.exit_code}")


def run_command_timed(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        end = time.perf_counter()
        return CommandResult(
            command=command,
            exit_code=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            elapsed_ms=(end - start) * 1000.0,
        )
    except OSError as exc:
        end = time.perf_counter()
        return CommandResult(
            command=command,
            exit_code=127,
            stdout="",
            stderr=str(exc),
            elapsed_ms=(end - start) * 1000.0,
        )


def summarize_failure(result: CommandResult, max_lines: int = 3, max_chars: int = 400) -> str:
    source = result.stderr.strip() if result.stderr.strip() else result.stdout.strip()
    if not source:
        return "no command output"
    lines = source.splitlines()
    summary = " | ".join(lines[:max_lines]).strip()
    if len(summary) > max_chars:
        return summary[: max_chars - 3] + "..."
    return summary


def run_stage_or_fail(
    stage_name: str,
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    failure_out: dict[str, object] | None = None,
) -> CommandResult | None:
    log_stage_start(stage_name)
    result = run_command_timed(command=command, cwd=cwd, env=env)
    log_stage_end(stage_name, result)
    if result.exit_code != 0:
        summary = summarize_failure(result)
        if failure_out is not None:
            failure_out["exit_code"] = int(result.exit_code)
            failure_out["summary"] = summary
        print(
            f"error: stage '{stage_name}' failed: {summary}",
            file=sys.stderr,
        )
        return None
    return result


def stage_result_row(stage_name: str, command: list[str], result: CommandResult) -> dict[str, object]:
    return {
        "stage": stage_name,
        "exit_code": int(result.exit_code),
        "latency_ms": round(result.elapsed_ms, 3),
        "command": " ".join(command),
    }


def manifest_paths(data_dir: Path) -> dict[str, Path]:
    clusters_current = data_dir / "clusters" / "current"
    return {
        "top": clusters_current / "cluster_manifest.bin",
        "mid": clusters_current / "mid_layer_clustering" / "MID_LAYER_CLUSTERING.bin",
        "lower": clusters_current / "lower_layer_clustering" / "LOWER_LAYER_CLUSTERING.bin",
        "final": clusters_current / "final_layer_clustering" / "FINAL_LAYER_CLUSTERS.bin",
    }


def read_manifest_bin(path: Path, validate_crc: bool = False) -> tuple[dict[str, int], bytes]:
    raw = path.read_bytes()
    if len(raw) < MANIFEST_HEADER_BYTES:
        raise ValueError(f"manifest too short: {path}")

    header = {
        "schema_version": int.from_bytes(raw[0:2], byteorder="little", signed=False),
        "record_type": int.from_bytes(raw[2:4], byteorder="little", signed=False),
        "record_count": int.from_bytes(raw[4:8], byteorder="little", signed=False),
        "payload_bytes": int.from_bytes(raw[8:12], byteorder="little", signed=False),
        "checksum_crc32": int.from_bytes(raw[12:16], byteorder="little", signed=False),
    }
    payload_end = MANIFEST_HEADER_BYTES + header["payload_bytes"]
    if len(raw) < payload_end:
        raise ValueError(f"manifest payload shorter than header declares: {path}")
    payload = raw[MANIFEST_HEADER_BYTES:payload_end]

    if validate_crc:
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != header["checksum_crc32"]:
            raise ValueError(f"manifest crc mismatch: {path}")

    return header, payload


def parse_manifest_payload(path: Path, validate_crc: bool = False) -> tuple[dict[str, int], dict[str, object]]:
    header, payload_bytes = read_manifest_bin(path=path, validate_crc=validate_crc)
    try:
        payload_text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"manifest payload is not valid utf-8: {path}") from exc
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest payload is not valid json: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest payload must be a json object: {path}")
    return header, payload


def _parse_ingest_inserted(ingest_result: CommandResult | None) -> int | None:
    if ingest_result is None:
        return None
    text = ingest_result.stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    inserted = payload.get("inserted")
    if isinstance(inserted, int) and inserted >= 0:
        return inserted
    return None


def compute_step5_summary(
    data_dir: Path,
    rows_written: int,
    pipeline_stage_results: list[dict[str, object]],
    ingest_result: CommandResult | None,
) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    paths = manifest_paths(data_dir)

    _, top_payload = parse_manifest_payload(paths["top"])
    _, mid_payload = parse_manifest_payload(paths["mid"])
    _, lower_payload = parse_manifest_payload(paths["lower"])
    final_header, final_payload = parse_manifest_payload(paths["final"])

    inserted = _parse_ingest_inserted(ingest_result)
    if inserted is None:
        inserted = int(rows_written)
        warnings.append("ingest inserted count unavailable; using rows_written fallback")

    top_cluster_count_obj = top_payload.get("chosen_k")
    if isinstance(top_cluster_count_obj, int) and top_cluster_count_obj >= 0:
        top_cluster_count = top_cluster_count_obj
    else:
        fallback = top_payload.get("record_count")
        if isinstance(fallback, int) and fallback >= 0:
            top_cluster_count = fallback
            warnings.append("top chosen_k missing; using top payload record_count fallback")
        else:
            top_cluster_count = 0
            warnings.append("top chosen_k missing and no top record_count fallback; using 0")

    parent_jobs = mid_payload.get("parent_jobs")
    mid_cluster_count = 0
    if isinstance(parent_jobs, list):
        if all(isinstance(j, dict) and isinstance(j.get("chosen_k"), int) and int(j.get("chosen_k")) >= 0 for j in parent_jobs):
            mid_cluster_count = sum(int(j.get("chosen_k")) for j in parent_jobs if isinstance(j, dict))
        else:
            mid_cluster_count = len(parent_jobs)
            warnings.append("mid parent_jobs chosen_k incomplete; using len(parent_jobs) fallback")
    else:
        warnings.append("mid parent_jobs missing; using 0")

    lower_continue_obj = lower_payload.get("branches_continue")
    lower_stop_obj = lower_payload.get("branches_stop")
    if isinstance(lower_continue_obj, int) and lower_continue_obj >= 0:
        lower_branches_continue = lower_continue_obj
    else:
        lower_branches_continue = 0
        warnings.append("lower branches_continue missing; using 0")
    if isinstance(lower_stop_obj, int) and lower_stop_obj >= 0:
        lower_branches_stop = lower_stop_obj
    else:
        lower_branches_stop = 0
        warnings.append("lower branches_stop missing; using 0")

    final_record_count_obj = final_payload.get("record_count")
    clusters_obj = final_payload.get("clusters")
    clusters_len: int | None = len(clusters_obj) if isinstance(clusters_obj, list) else None
    if isinstance(final_record_count_obj, int) and final_record_count_obj >= 0:
        final_cluster_count = final_record_count_obj
        if clusters_len is not None and clusters_len != final_cluster_count:
            warnings.append("final record_count does not match len(clusters); using record_count")
    elif clusters_len is not None:
        final_cluster_count = clusters_len
        warnings.append("final record_count missing; using len(clusters) fallback")
    else:
        final_cluster_count = int(final_header.get("record_count", 0))
        warnings.append("final payload missing record_count and clusters; using header record_count fallback")

    total_pipeline_latency_ms = 0.0
    for row in pipeline_stage_results:
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)):
            total_pipeline_latency_ms += float(latency)

    summary = {
        "total_embeddings_inserted": int(inserted),
        "top_cluster_count": int(top_cluster_count),
        "mid_cluster_count": int(mid_cluster_count),
        "lower_branches_continue": int(lower_branches_continue),
        "lower_branches_stop": int(lower_branches_stop),
        "final_cluster_count": int(final_cluster_count),
        "total_pipeline_latency_ms": round(total_pipeline_latency_ms, 3),
    }
    return summary, warnings


def print_step5_summary(summary: dict[str, object]) -> None:
    print("=== Cluster Summary ===")
    print(f"total_embeddings_inserted: {summary['total_embeddings_inserted']}")
    print(f"top_cluster_count: {summary['top_cluster_count']}")
    print(f"mid_cluster_count: {summary['mid_cluster_count']}")
    print(f"lower_branches_continue: {summary['lower_branches_continue']}")
    print(f"lower_branches_stop: {summary['lower_branches_stop']}")
    print(f"final_cluster_count: {summary['final_cluster_count']}")
    print(f"total_pipeline_latency_ms: {float(summary['total_pipeline_latency_ms']):.3f}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_counts_summary(rows_written: int | None, cluster_summary: dict[str, object] | None, ingest_result: CommandResult | None) -> dict[str, int | None]:
    inserted = _parse_ingest_inserted(ingest_result)
    if inserted is None and rows_written is not None:
        inserted = int(rows_written)
    return {
        "inserted": int(inserted) if inserted is not None else None,
        "top": int(cluster_summary["top_cluster_count"]) if cluster_summary and "top_cluster_count" in cluster_summary else None,
        "mid": int(cluster_summary["mid_cluster_count"]) if cluster_summary and "mid_cluster_count" in cluster_summary else None,
        "lower_continue": int(cluster_summary["lower_branches_continue"]) if cluster_summary and "lower_branches_continue" in cluster_summary else None,
        "lower_stop": int(cluster_summary["lower_branches_stop"]) if cluster_summary and "lower_branches_stop" in cluster_summary else None,
        "final": int(cluster_summary["final_cluster_count"]) if cluster_summary and "final_cluster_count" in cluster_summary else None,
    }


def build_result_payload(
    *,
    status: str,
    failure_detail: dict[str, object] | None,
    exit_code: int,
    args_snapshot: dict[str, object],
    seed_mode: str,
    runner_smoke_stage_results: list[dict[str, object]],
    pipeline_stage_results: list[dict[str, object]],
    cluster_summary: dict[str, object] | None,
    cluster_summary_warnings: list[str],
    counts_summary: dict[str, int | None],
    build_dir: Path | None,
    cli_binary: Path | None,
    data_dir: Path | None,
    dataset_path: Path | None,
    results_out: Path | None,
    rows_written: int | None,
    run_full_pipeline: bool,
    with_search_sanity: bool,
    with_cluster_stats: bool,
    keep_data: bool,
    input_format: str | None,
) -> dict[str, object]:
    return {
        "status": status,
        "failure_detail": failure_detail,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "args": args_snapshot,
        "seed_mode": seed_mode,
        "stage_results": {
            "runner_smoke": runner_smoke_stage_results,
            "full_pipeline": pipeline_stage_results,
        },
        "cluster_summary": cluster_summary,
        "cluster_summary_warnings": cluster_summary_warnings,
        "counts_summary": counts_summary,
        "artifacts": {
            "build_dir": str(build_dir) if build_dir is not None else None,
            "cli_binary": str(cli_binary) if cli_binary is not None else None,
            "data_dir": str(data_dir) if data_dir is not None else None,
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "results_out": str(results_out) if results_out is not None else None,
        },
        "exit_code": int(exit_code),
        "command": "pipeline-test-generate",
        "input_format": input_format,
        "rows_written": rows_written,
        "run_full_pipeline": run_full_pipeline,
        "with_search_sanity": with_search_sanity,
        "with_cluster_stats": with_cluster_stats,
        "keep_data": keep_data,
    }


def resolve_cli_binary(build_dir: Path) -> Path | None:
    candidates = [build_dir / "vectordb_v3_cli.exe", build_dir / "vectordb_v3_cli"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic dataset generator scaffold for vector_db_v3 pipeline testing.",
    )
    parser.add_argument(
        "--build-dir",
        required=True,
        help="Path to build directory containing vectordb_v3_cli(.exe).",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional data directory for future pipeline artifacts.",
    )
    parser.add_argument(
        "--embedding-count",
        required=True,
        type=int,
        help="Required embedding count (>0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for future bulk ingest path (>0).",
    )
    parser.add_argument(
        "--results-out",
        default="",
        help="Optional output JSON path; default is <data-dir>/pipeline_test_results.json.",
    )
    parser.add_argument(
        "--input-format",
        choices=["bin", "jsonl"],
        default="bin",
        help="Dataset output format (default: bin).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. If omitted, uses entropy random per run.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Preserve test data directory after future pipeline runs.",
    )
    parser.add_argument(
        "--runner-smoke",
        action="store_true",
        help="Run Step-3 smoke command stages (init, stats) with lifecycle latency logging.",
    )
    parser.add_argument(
        "--run-full-pipeline",
        action="store_true",
        help="Run full Step-4 ordered pipeline stages after dataset generation.",
    )
    parser.add_argument(
        "--with-search-sanity",
        action="store_true",
        help="Append optional search sanity stage when --run-full-pipeline is enabled.",
    )
    parser.add_argument(
        "--with-cluster-stats",
        action="store_true",
        help="Append optional cluster-stats stage when --run-full-pipeline is enabled.",
    )
    return parser


def resolve_dataset_path(data_dir: Path, input_format: str) -> Path:
    if input_format == "jsonl":
        return data_dir / "bulk.jsonl"
    if input_format == "bin":
        return data_dir / "bulk.bin"
    raise ValueError(f"unsupported input format: {input_format}")


def random_vector(dim: int, rng: random.Random) -> list[float]:
    # Use float32-friendly range [0,1) and keep values stable for downstream parsing.
    return [rng.random() for _ in range(dim)]


def write_query_vec_file(path: Path, dim: int = VECTOR_DIM) -> None:
    values = ",".join("0.0" for _ in range(dim))
    path.write_text(values, encoding="utf-8")


def write_synthetic_jsonl(path: Path, embedding_count: int, dim: int = VECTOR_DIM, seed: int | None = None) -> int:
    rng = random.Random(seed) if seed is not None else random.Random()
    if seed is None:
        rng.seed()
    rows_written = 0
    with path.open("w", encoding="utf-8") as f:
        for embedding_id in range(1, embedding_count + 1):
            row = {
                "embedding_id": embedding_id,
                "vector": random_vector(dim, rng),
            }
            f.write(json.dumps(row) + "\n")
            rows_written += 1
    return rows_written


def write_synthetic_bin(path: Path, embedding_count: int, dim: int = VECTOR_DIM, seed: int | None = None) -> int:
    if dim != VECTOR_DIM:
        raise ValueError("binary generator requires dim=1024")
    rng = random.Random(seed) if seed is not None else random.Random()
    if seed is None:
        rng.seed()

    rows_written = 0
    with path.open("wb") as f:
        header = (
            BIN_MAGIC_V3BI.to_bytes(4, byteorder="little", signed=False)
            + BIN_VERSION.to_bytes(2, byteorder="little", signed=False)
            + BIN_RECORD_BYTES.to_bytes(4, byteorder="little", signed=False)
            + embedding_count.to_bytes(8, byteorder="little", signed=False)
        )
        f.write(header)

        for embedding_id in range(1, embedding_count + 1):
            f.write(int(embedding_id).to_bytes(8, byteorder="little", signed=False))
            for value in random_vector(dim, rng):
                f.write(struct.pack("<f", float(value)))
            rows_written += 1
    return rows_written


def validate_generated_file(
    path: Path,
    input_format: str,
    expected_count: int,
    dim: int = VECTOR_DIM,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "generated dataset file not found"

    if input_format == "jsonl":
        count = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                    except Exception:
                        return False, f"jsonl parse failed at line {line_no}"
                    if "embedding_id" not in obj or "vector" not in obj:
                        return False, f"jsonl row missing required fields at line {line_no}"
                    if not isinstance(obj["vector"], list) or len(obj["vector"]) != dim:
                        return False, f"jsonl vector dim mismatch at line {line_no}"
                    count += 1
        except OSError as exc:
            return False, f"unable to read generated jsonl: {exc}"
        if count != expected_count:
            return False, f"jsonl row count mismatch: expected {expected_count}, got {count}"
        return True, "ok"

    if input_format == "bin":
        try:
            size = path.stat().st_size
        except OSError as exc:
            return False, f"unable to inspect generated bin file: {exc}"
        if size < BIN_HEADER_BYTES:
            return False, "binary file too short for header"
        try:
            with path.open("rb") as f:
                header = f.read(BIN_HEADER_BYTES)
                if len(header) != BIN_HEADER_BYTES:
                    return False, "binary header read failed"
                magic = int.from_bytes(header[0:4], byteorder="little", signed=False)
                version = int.from_bytes(header[4:6], byteorder="little", signed=False)
                record_size = int.from_bytes(header[6:10], byteorder="little", signed=False)
                record_count = int.from_bytes(header[10:18], byteorder="little", signed=False)
        except OSError as exc:
            return False, f"unable to read generated bin file: {exc}"
        if magic != BIN_MAGIC_V3BI:
            return False, "binary magic mismatch"
        if version != BIN_VERSION:
            return False, "binary version mismatch"
        expected_record_bytes = 8 + dim * 4
        if record_size != expected_record_bytes:
            return False, "binary record size mismatch"
        if record_count != expected_count:
            return False, f"binary record_count mismatch: expected {expected_count}, got {record_count}"
        payload_size = size - BIN_HEADER_BYTES
        expected_payload_size = expected_count * expected_record_bytes
        if payload_size != expected_payload_size:
            return False, "binary payload size mismatch"
        return True, "ok"

    return False, "unsupported input format for validation"


def build_pipeline_stages(
    cli_binary: Path,
    data_dir: Path,
    dataset_path: Path,
    input_format: str,
    batch_size: int,
    with_search_sanity: bool,
    with_cluster_stats: bool,
    query_vec_path: Path | None,
) -> list[tuple[str, list[str]]]:
    if input_format == "jsonl":
        ingest_cmd = [
            str(cli_binary),
            "bulk-insert",
            "--path",
            str(data_dir),
            "--input",
            str(dataset_path),
            "--batch-size",
            str(batch_size),
        ]
    elif input_format == "bin":
        ingest_cmd = [
            str(cli_binary),
            "bulk-insert-bin",
            "--path",
            str(data_dir),
            "--input",
            str(dataset_path),
            "--batch-size",
            str(batch_size),
        ]
    else:
        raise ValueError(f"unsupported input format: {input_format}")

    stages: list[tuple[str, list[str]]] = [
        ("init", [str(cli_binary), "init", "--path", str(data_dir)]),
        ("ingest", ingest_cmd),
        ("build-top-clusters", [str(cli_binary), "build-top-clusters", "--path", str(data_dir)]),
        ("build-mid-layer-clusters", [str(cli_binary), "build-mid-layer-clusters", "--path", str(data_dir)]),
        ("build-lower-layer-clusters", [str(cli_binary), "build-lower-layer-clusters", "--path", str(data_dir)]),
        ("build-final-layer-clusters", [str(cli_binary), "build-final-layer-clusters", "--path", str(data_dir)]),
    ]

    if with_search_sanity:
        if query_vec_path is None:
            raise ValueError("query vector path is required when with_search_sanity is enabled")
        stages.append(
            (
                "search",
                [
                    str(cli_binary),
                    "search",
                    "--path",
                    str(data_dir),
                    "--vec",
                    str(query_vec_path),
                    "--topk",
                    "5",
                ],
            )
        )
    if with_cluster_stats:
        stages.append(("cluster-stats", [str(cli_binary), "cluster-stats", "--path", str(data_dir)]))

    return stages


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    status = "pass"
    exit_code = 0
    failure_detail: dict[str, object] | None = None

    build_dir: Path | None = Path(args.build_dir).resolve() if args.build_dir else None
    cli_binary: Path | None = None
    if args.data_dir:
        data_dir: Path | None = Path(args.data_dir).resolve()
    else:
        data_dir = Path(tempfile.gettempdir()) / "vectordb_v3_pipeline_test"
    if args.results_out:
        results_out: Path | None = Path(args.results_out).resolve()
    else:
        results_out = (data_dir / "pipeline_test_results.json") if data_dir is not None else None
    dataset_path: Path | None = None
    rows_written: int | None = None

    smoke_stage_results: list[dict[str, object]] = []
    pipeline_stage_results: list[dict[str, object]] = []
    ingest_stage_result: CommandResult | None = None
    cluster_summary: dict[str, object] | None = None
    cluster_summary_warnings: list[str] = []
    stage_failure_ctx: dict[str, object] | None = None

    def set_failure(
        *,
        kind: str,
        message: str,
        code: int,
        stage: str | None = None,
        command: str | None = None,
        stage_exit_code: int | None = None,
        summary: str | None = None,
    ) -> None:
        nonlocal status, exit_code, failure_detail
        if failure_detail is not None:
            return
        status = "fail"
        exit_code = code
        failure_detail = {
            "kind": kind,
            "message": message,
            "stage": stage,
            "command": command,
            "exit_code": stage_exit_code,
            "summary": summary,
        }
        print(f"error: {message}", file=sys.stderr)

    if args.embedding_count <= 0:
        set_failure(kind="usage", message="--embedding-count must be > 0", code=2)
    if failure_detail is None and args.batch_size <= 0:
        set_failure(kind="usage", message="--batch-size must be > 0", code=2)
    if failure_detail is None and args.runner_smoke and args.run_full_pipeline:
        set_failure(kind="usage", message="--runner-smoke cannot be combined with --run-full-pipeline", code=2)
    if failure_detail is None and (args.with_search_sanity or args.with_cluster_stats) and not args.run_full_pipeline:
        set_failure(
            kind="usage",
            message="--with-search-sanity/--with-cluster-stats require --run-full-pipeline",
            code=2,
        )

    if failure_detail is None:
        cli_binary = resolve_cli_binary(build_dir) if build_dir is not None else None
        if cli_binary is None:
            set_failure(
                kind="usage",
                message=f"missing vectordb_v3_cli binary in --build-dir: {build_dir}",
                code=2,
            )

    if failure_detail is None and data_dir is not None:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            set_failure(kind="runtime", message=f"failed to create data directory: {exc}", code=1)

    if failure_detail is None and data_dir is not None:
        try:
            dataset_path = resolve_dataset_path(data_dir, args.input_format)
        except ValueError as exc:
            set_failure(kind="usage", message=str(exc), code=2)

    if failure_detail is None and dataset_path is not None:
        try:
            if args.input_format == "jsonl":
                rows_written = write_synthetic_jsonl(
                    path=dataset_path,
                    embedding_count=args.embedding_count,
                    dim=VECTOR_DIM,
                    seed=args.seed,
                )
            else:
                rows_written = write_synthetic_bin(
                    path=dataset_path,
                    embedding_count=args.embedding_count,
                    dim=VECTOR_DIM,
                    seed=args.seed,
                )
        except (OSError, ValueError) as exc:
            set_failure(kind="runtime", message=f"failed to generate synthetic dataset: {exc}", code=1)

    if failure_detail is None and dataset_path is not None:
        ok, message = validate_generated_file(
            path=dataset_path,
            input_format=args.input_format,
            expected_count=args.embedding_count,
            dim=VECTOR_DIM,
        )
        if not ok:
            set_failure(kind="runtime", message=f"generated dataset validation failed: {message}", code=1)

    if failure_detail is None and args.runner_smoke and cli_binary is not None and data_dir is not None:
        smoke_stages = [
            ("init", [str(cli_binary), "init", "--path", str(data_dir)]),
            ("stats", [str(cli_binary), "stats", "--path", str(data_dir)]),
        ]
        for stage_name, stage_command in smoke_stages:
            stage_failure_ctx = {}
            stage_result = run_stage_or_fail(
                stage_name=stage_name,
                command=stage_command,
                failure_out=stage_failure_ctx,
            )
            if stage_result is None:
                set_failure(
                    kind="stage_failure",
                    message=f"stage '{stage_name}' failed",
                    code=1,
                    stage=stage_name,
                    command=" ".join(stage_command),
                    stage_exit_code=stage_failure_ctx.get("exit_code") if isinstance(stage_failure_ctx.get("exit_code"), int) else None,
                    summary=stage_failure_ctx.get("summary") if isinstance(stage_failure_ctx.get("summary"), str) else None,
                )
                break
            smoke_stage_results.append(stage_result_row(stage_name, stage_command, stage_result))

    if failure_detail is None and args.run_full_pipeline and cli_binary is not None and data_dir is not None and dataset_path is not None:
        query_vec_path: Path | None = None
        if args.with_search_sanity:
            query_vec_path = data_dir / "search_query.vec"
            try:
                write_query_vec_file(query_vec_path, dim=VECTOR_DIM)
            except OSError as exc:
                set_failure(kind="runtime", message=f"failed to write search sanity vector: {exc}", code=1)
        if failure_detail is None:
            try:
                full_stages = build_pipeline_stages(
                    cli_binary=cli_binary,
                    data_dir=data_dir,
                    dataset_path=dataset_path,
                    input_format=args.input_format,
                    batch_size=int(args.batch_size),
                    with_search_sanity=bool(args.with_search_sanity),
                    with_cluster_stats=bool(args.with_cluster_stats),
                    query_vec_path=query_vec_path,
                )
            except ValueError as exc:
                set_failure(kind="usage", message=str(exc), code=2)
                full_stages = []

            for stage_name, stage_command in full_stages:
                if failure_detail is not None:
                    break
                stage_failure_ctx = {}
                stage_result = run_stage_or_fail(
                    stage_name=stage_name,
                    command=stage_command,
                    failure_out=stage_failure_ctx,
                )
                if stage_result is None:
                    set_failure(
                        kind="stage_failure",
                        message=f"stage '{stage_name}' failed",
                        code=1,
                        stage=stage_name,
                        command=" ".join(stage_command),
                        stage_exit_code=stage_failure_ctx.get("exit_code") if isinstance(stage_failure_ctx.get("exit_code"), int) else None,
                        summary=stage_failure_ctx.get("summary") if isinstance(stage_failure_ctx.get("summary"), str) else None,
                    )
                    break
                pipeline_stage_results.append(stage_result_row(stage_name, stage_command, stage_result))
                if stage_name == "ingest":
                    ingest_stage_result = stage_result

        if failure_detail is None:
            try:
                cluster_summary, cluster_summary_warnings = compute_step5_summary(
                    data_dir=data_dir,
                    rows_written=rows_written or 0,
                    pipeline_stage_results=pipeline_stage_results,
                    ingest_result=ingest_stage_result,
                )
            except (OSError, ValueError) as exc:
                set_failure(
                    kind="runtime",
                    message=f"failed to parse cluster summary manifests: {exc}",
                    code=1,
                )
            if failure_detail is None and cluster_summary is not None:
                for warning in cluster_summary_warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                print_step5_summary(cluster_summary)

    counts_summary = build_counts_summary(rows_written, cluster_summary, ingest_stage_result)
    args_snapshot = {
        "build_dir": str(build_dir) if build_dir is not None else None,
        "data_dir": str(data_dir) if data_dir is not None else None,
        "embedding_count": int(args.embedding_count),
        "batch_size": int(args.batch_size),
        "input_format": args.input_format,
        "run_full_pipeline": bool(args.run_full_pipeline),
        "runner_smoke": bool(args.runner_smoke),
        "with_search_sanity": bool(args.with_search_sanity),
        "with_cluster_stats": bool(args.with_cluster_stats),
        "seed": args.seed,
        "results_out": str(results_out) if results_out is not None else None,
        "keep_data": bool(args.keep_data),
    }
    seed_mode = "deterministic" if args.seed is not None else "entropy"

    payload = build_result_payload(
        status=status,
        failure_detail=failure_detail,
        exit_code=exit_code,
        args_snapshot=args_snapshot,
        seed_mode=seed_mode,
        runner_smoke_stage_results=smoke_stage_results,
        pipeline_stage_results=pipeline_stage_results,
        cluster_summary=cluster_summary,
        cluster_summary_warnings=cluster_summary_warnings,
        counts_summary=counts_summary,
        build_dir=build_dir,
        cli_binary=cli_binary,
        data_dir=data_dir,
        dataset_path=dataset_path,
        results_out=results_out,
        rows_written=rows_written,
        run_full_pipeline=bool(args.run_full_pipeline),
        with_search_sanity=bool(args.with_search_sanity),
        with_cluster_stats=bool(args.with_cluster_stats),
        keep_data=bool(args.keep_data),
        input_format=args.input_format,
    )

    if results_out is not None:
        try:
            write_json(results_out, payload)
        except OSError as exc:
            print(f"error: failed to write results json: {exc}", file=sys.stderr)
            if failure_detail is None:
                status = "fail"
                exit_code = 1
                failure_detail = {
                    "kind": "write_failure",
                    "message": f"failed to write results json: {exc}",
                    "stage": None,
                    "command": None,
                    "exit_code": None,
                    "summary": None,
                }
                payload = build_result_payload(
                    status=status,
                    failure_detail=failure_detail,
                    exit_code=exit_code,
                    args_snapshot=args_snapshot,
                    seed_mode=seed_mode,
                    runner_smoke_stage_results=smoke_stage_results,
                    pipeline_stage_results=pipeline_stage_results,
                    cluster_summary=cluster_summary,
                    cluster_summary_warnings=cluster_summary_warnings,
                    counts_summary=counts_summary,
                    build_dir=build_dir,
                    cli_binary=cli_binary,
                    data_dir=data_dir,
                    dataset_path=dataset_path,
                    results_out=results_out,
                    rows_written=rows_written,
                    run_full_pipeline=bool(args.run_full_pipeline),
                    with_search_sanity=bool(args.with_search_sanity),
                    with_cluster_stats=bool(args.with_cluster_stats),
                    keep_data=bool(args.keep_data),
                    input_format=args.input_format,
                )

    print(json.dumps(payload, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

