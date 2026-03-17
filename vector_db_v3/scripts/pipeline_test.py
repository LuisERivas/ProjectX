from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import tempfile
from pathlib import Path

VECTOR_DIM = 1024
BIN_MAGIC_V3BI = 0x49423356
BIN_VERSION = 1
BIN_HEADER_BYTES = 18  # u32 magic + u16 version + u32 record_size + u64 record_count
BIN_RECORD_BYTES = 8 + VECTOR_DIM * 4  # u64 embedding_id + 1024 * f32


def emit_usage_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def emit_runtime_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.embedding_count <= 0:
        return emit_usage_error("--embedding-count must be > 0")
    if args.batch_size <= 0:
        return emit_usage_error("--batch-size must be > 0")

    build_dir = Path(args.build_dir).resolve()
    cli_binary = resolve_cli_binary(build_dir)
    if cli_binary is None:
        return emit_usage_error(f"missing vectordb_v3_cli binary in --build-dir: {build_dir}")

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = Path(tempfile.gettempdir()) / "vectordb_v3_pipeline_test"

    if args.results_out:
        results_out = Path(args.results_out).resolve()
    else:
        results_out = data_dir / "pipeline_test_results.json"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return emit_runtime_error(f"failed to create data directory: {exc}")

    try:
        dataset_path = resolve_dataset_path(data_dir, args.input_format)
    except ValueError as exc:
        return emit_usage_error(str(exc))

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
        return emit_runtime_error(f"failed to generate synthetic dataset: {exc}")

    ok, message = validate_generated_file(
        path=dataset_path,
        input_format=args.input_format,
        expected_count=args.embedding_count,
        dim=VECTOR_DIM,
    )
    if not ok:
        return emit_runtime_error(f"generated dataset validation failed: {message}")

    payload = {
        "status": "ok",
        "command": "pipeline-test-generate",
        "build_dir": str(build_dir),
        "cli_binary": str(cli_binary),
        "data_dir": str(data_dir),
        "embedding_count": int(args.embedding_count),
        "batch_size": int(args.batch_size),
        "input_format": args.input_format,
        "seed_mode": "deterministic" if args.seed is not None else "entropy",
        "dataset_path": str(dataset_path),
        "rows_written": rows_written,
        "results_out": str(results_out),
        "keep_data": bool(args.keep_data),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

