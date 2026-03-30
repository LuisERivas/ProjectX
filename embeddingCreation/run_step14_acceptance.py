#!/usr/bin/env python3
"""
Step 14 acceptance runner.

Runs the full real pipeline on a fixed corpus and asserts final requirements.
Exit code:
  0 => acceptance passed
  1 => acceptance failed
  2 => runtime requirements missing (CUDA/PyICU/torch)
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - handled by requirement gate
    np = None

EXPECTED_MODEL_ID = "voyageai/voyage-4-nano"
EXPECTED_RECORD_SIZE = 4100
EXPECTED_DIM = 2048
UINT32_MAX = (2**32) - 1
SPOTCHECK_SEED = 42
SPOTCHECK_COUNT = 5


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@dataclass
class _AssertionState:
    passed: int = 0
    failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        default="fixtures/acceptance_corpus",
        help="path to acceptance corpus directory (default: fixtures/acceptance_corpus)",
    )
    parser.add_argument(
        "--workspace",
        default="tmp_step14_acceptance",
        help="temporary workspace for acceptance output (default: tmp_step14_acceptance)",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="keep generated output file and workspace",
    )
    return parser.parse_args()


def _check_requirements() -> tuple[bool, str | None]:
    if np is None:
        return False, "NumPy unavailable"

    try:
        import icu  # noqa: F401
    except Exception as exc:
        return False, f"PyICU unavailable: {exc}"

    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch unavailable: {exc}"

    if not torch.cuda.is_available():
        return False, "CUDA unavailable (torch.cuda.is_available() == False)"
    return True, None


def _assert(state: _AssertionState, ok: bool, description: str, detail: str = "") -> None:
    if ok:
        state.passed += 1
        print(f"PASS: {description}")
    else:
        state.failed += 1
        suffix = f" ({detail})" if detail else ""
        print(f"FAIL: {description}{suffix}")


def _count_log(records: list[logging.LogRecord], needle: str) -> int:
    n = 0
    for rec in records:
        try:
            msg = rec.getMessage()
        except Exception:
            continue
        if needle in msg:
            n += 1
    return n


def _run_acceptance(corpus_dir: Path, output_path: Path) -> _AssertionState:
    from binary_reader import read_all, read_record, record_count, verify_file
    from embedding_worker import MODEL_ID
    from ingest_pipeline import run_pipeline

    state = _AssertionState()

    # 1) model id constant
    _assert(
        state,
        MODEL_ID == EXPECTED_MODEL_ID,
        "embedding_worker MODEL_ID matches voyage-4-nano contract",
        f"got={MODEL_ID}",
    )

    # 2) icu import
    try:
        import icu  # noqa: F401

        icu_ok = True
    except Exception as exc:
        icu_ok = False
        icu_err = str(exc)
    _assert(state, icu_ok, "PyICU import succeeds", "" if icu_ok else icu_err)

    # 3) sentence_splitter uses ICU BreakIterator
    try:
        import sentence_splitter

        uses_icu = hasattr(sentence_splitter, "icu") and hasattr(sentence_splitter.icu, "BreakIterator")
    except Exception:
        uses_icu = False
    _assert(state, uses_icu, "sentence_splitter uses ICU BreakIterator")

    worker_log_handler = _ListHandler()
    ingest_log_handler = _ListHandler()
    worker_logger = logging.getLogger("embedding_worker")
    ingest_logger = logging.getLogger("ingest_pipeline")
    worker_logger.addHandler(worker_log_handler)
    ingest_logger.addHandler(ingest_log_handler)
    worker_logger.setLevel(logging.INFO)
    ingest_logger.setLevel(logging.INFO)

    try:
        result = run_pipeline(corpus_dir, output_path, locale="en_US")
    finally:
        worker_logger.removeHandler(worker_log_handler)
        ingest_logger.removeHandler(ingest_log_handler)

    # 4) pipeline result success
    _assert(state, result.success is True, "PipelineResult.success is True", f"errors={result.errors}")

    # 5/6/7 files counters
    _assert(state, result.files_discovered == 4, "files_discovered equals 4", str(result.files_discovered))
    _assert(state, result.files_read == 4, "files_read equals 4", str(result.files_read))
    _assert(state, result.files_skipped == 0, "files_skipped equals 0", str(result.files_skipped))

    # 8/9 sentence/record counters
    _assert(
        state,
        result.total_sentences >= 10,
        "total_sentences is at least 10",
        str(result.total_sentences),
    )
    _assert(
        state,
        result.total_sentences == result.records_written,
        "total_sentences equals records_written",
        f"{result.total_sentences} vs {result.records_written}",
    )

    # 10/11/12 output and size/count checks
    _assert(state, output_path.exists(), "output binary exists", str(output_path))
    if output_path.exists():
        size = output_path.stat().st_size
    else:
        size = -1
    _assert(
        state,
        size == result.records_written * EXPECTED_RECORD_SIZE,
        "output file size equals records_written * 4100",
        f"{size} vs {result.records_written * EXPECTED_RECORD_SIZE}",
    )
    try:
        rc = record_count(output_path)
        rc_ok = True
    except Exception as exc:
        rc_ok = False
        rc = -1
        rc_err = str(exc)
    _assert(
        state,
        rc_ok and rc == result.records_written,
        "record_count(output) equals records_written",
        rc_err if not rc_ok else f"{rc} vs {result.records_written}",
    )

    # 13..19 verification checks
    try:
        report = verify_file(output_path)
    except Exception as exc:
        report = None
        report_err = str(exc)

    _assert(state, report is not None and report.ok, "verify_file(output).ok is True", report_err if report is None else str(report.errors))
    _assert(
        state,
        report is not None and report.norms_all_in_tolerance,
        "verify_file(output).norms_all_in_tolerance is True",
    )
    _assert(state, report is not None and report.all_finite, "verify_file(output).all_finite is True")
    _assert(state, report is not None and report.ids_monotonic, "verify_file(output).ids_monotonic is True")
    _assert(state, report is not None and report.ids_unique, "verify_file(output).ids_unique is True")
    _assert(
        state,
        report is not None and report.id_min is not None and report.id_min >= 0,
        "verify_file(output).id_min is >= 0",
        str(None if report is None else report.id_min),
    )
    _assert(
        state,
        report is not None and report.id_max is not None and report.id_max <= UINT32_MAX,
        "verify_file(output).id_max is <= uint32 max",
        str(None if report is None else report.id_max),
    )

    # 20/21 dtype and dim checks via read_all
    ids, embs = read_all(output_path)
    _assert(state, embs.dtype == np.float16, "read_all(output) dtype is float16", str(embs.dtype))
    _assert(
        state,
        embs.ndim == 2 and embs.shape[1] == EXPECTED_DIM,
        "read_all(output) second dimension is 2048",
        str(embs.shape),
    )

    # 22 random spot checks
    rng = random.Random(SPOTCHECK_SEED)
    all_spot_ok = True
    if len(ids) == 0:
        all_spot_ok = False
    else:
        k = min(SPOTCHECK_COUNT, len(ids))
        indices = rng.sample(range(len(ids)), k=k)
        for idx in indices:
            rid, row = read_record(output_path, idx)
            if row.shape[0] != EXPECTED_DIM:
                all_spot_ok = False
                break
            norm = float(np.linalg.norm(row.astype(np.float32)))
            if not (0.95 <= norm <= 1.05):
                all_spot_ok = False
                break
            if not (0 <= rid <= UINT32_MAX):
                all_spot_ok = False
                break
    _assert(state, all_spot_ok, "random spot checks pass (seed=42, 5 records)")

    # 23/24 load once + explicit shutdown by logs
    init_count = _count_log(worker_log_handler.records, "embedding worker initialized")
    shutdown_count = _count_log(worker_log_handler.records, "model unloaded and worker shut down")
    _assert(
        state,
        init_count == 1,
        'logs contain exactly 1 "embedding worker initialized"',
        f"count={init_count}",
    )
    _assert(
        state,
        shutdown_count == 1,
        'logs contain exactly 1 "model unloaded and worker shut down"',
        f"count={shutdown_count}",
    )

    # 25 result.verification present
    _assert(state, result.verification is not None, "PipelineResult.verification is not None")
    return state


def main() -> int:
    args = parse_args()
    ok, reason = _check_requirements()
    if not ok:
        print("acceptance_skipped: missing runtime requirements", file=sys.stderr)
        print(f"details: {reason}", file=sys.stderr)
        return 2

    corpus_dir = Path(args.corpus_dir)
    workspace = Path(args.workspace)
    output_path = workspace / "step14_acceptance.bin"

    workspace.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    try:
        state = _run_acceptance(corpus_dir=corpus_dir, output_path=output_path)
        total = state.passed + state.failed
        if state.failed == 0:
            print(f"acceptance: PASSED ({state.passed}/{total})")
            return 0
        print(f"acceptance: FAILED ({state.passed}/{total} passed)")
        return 1
    finally:
        if args.keep_outputs:
            print(f"cleanup: skipped (--keep-outputs), workspace={workspace.resolve()}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)
            print("cleanup: ok")


if __name__ == "__main__":
    raise SystemExit(main())
