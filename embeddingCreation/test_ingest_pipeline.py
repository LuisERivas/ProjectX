#!/usr/bin/env python3
"""
Step 11 tests for end-to-end pipeline orchestration.

Usage:
  python3 test_ingest_pipeline.py
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from binary_writer import BinaryWriterError
from binary_reader import read_all
from embedding_worker import EmbeddingError, ModelLoadError
from packed_id import SentenceMeta
from functools import wraps

from ingest_pipeline import (
    DEFAULT_CHAR_LEN_BUCKET_EDGES,
    ProbeStrategy,
    _resolved_probe_candidates,
    _split_sorted_metas_by_char_len_edges,
    _validate_char_len_bucket_edges,
    probe_batch_size,
    run_pipeline,
)
import ingest_pipeline as ingest_pipeline_mod

try:
    import icu  # type: ignore

    HAVE_ICU = True
except Exception:
    HAVE_ICU = False


def _simple_splitter(text: str, *, locale: str) -> list[str]:
    del locale
    chunks = [c.strip() for c in text.replace("\n", " ").split(".")]
    return [c + "." for c in chunks if c]


class _FakeWorker:
    init_calls = 0
    shutdown_calls = 0
    encode_calls = 0
    fail_on_encode = False
    fail_on_batch_size_at_or_above: int | None = None
    seen_batch_sizes: list[int] = []
    seen_batches: list[list[str]] = []
    last_instance: "_FakeWorker | None" = None

    def __init__(self) -> None:
        self.state = "UNINITIALIZED"
        self.load_count = 0
        _FakeWorker.last_instance = self

    @classmethod
    def reset(cls) -> None:
        cls.init_calls = 0
        cls.shutdown_calls = 0
        cls.encode_calls = 0
        cls.fail_on_encode = False
        cls.fail_on_batch_size_at_or_above = None
        cls.seen_batch_sizes = []
        cls.seen_batches = []
        cls.last_instance = None

    @property
    def stats(self) -> dict[str, object]:
        return {"state": self.state, "load_count": self.load_count}

    def init(self) -> None:
        _FakeWorker.init_calls += 1
        self.state = "READY"
        self.load_count += 1

    def encode_batch(self, sentences: list[str]) -> np.ndarray:
        _FakeWorker.encode_calls += 1
        _FakeWorker.seen_batch_sizes.append(len(sentences))
        _FakeWorker.seen_batches.append(list(sentences))
        if _FakeWorker.fail_on_encode:
            raise EmbeddingError("forced encode failure")
        if (
            _FakeWorker.fail_on_batch_size_at_or_above is not None
            and len(sentences) >= _FakeWorker.fail_on_batch_size_at_or_above
        ):
            raise EmbeddingError("forced probe OOM")
        arr = np.zeros((len(sentences), 2048), dtype=np.float16)
        for i in range(len(sentences)):
            arr[i, i % 2048] = np.float16(1.0)
        return arr

    def shutdown(self) -> None:
        _FakeWorker.shutdown_calls += 1
        self.state = "SHUTDOWN"


class TestIngestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        _FakeWorker.reset()
        # Most tests patch the worker; bypass startup CUDA checks by default.
        self._startup_patcher = patch("ingest_pipeline._validate_startup", return_value=None)
        self._startup_patcher.start()
        self.addCleanup(self._startup_patcher.stop)

    def _write_texts(self, root: Path, files: dict[str, str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")

    def test_end_to_end_two_files(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B.", "b.txt": "C. D. E."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out, batch_size=4)
            self.assertTrue(out.exists())
            self.assertEqual(res.total_sentences, 5)
            self.assertEqual(res.records_written, 5)
            self.assertEqual(out.stat().st_size, 5 * 4106)

    def test_end_to_end_five_files(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(
                root,
                {
                    "a.txt": "A.",
                    "b.txt": "B. C.",
                    "c.txt": "D.",
                    "d.txt": "E. F.",
                    "e.txt": "G.",
                },
            )
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out, batch_size=8)
            self.assertEqual(res.total_sentences, 7)
            self.assertEqual(res.records_written, 7)
            self.assertTrue(res.verification and res.verification.ok)

    def test_sentences_equal_records(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B. C."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out, batch_size=2)
            self.assertEqual(res.total_sentences, res.records_written)

    def test_model_loads_once_and_shutdown_once(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                run_pipeline(root, out)
            self.assertEqual(_FakeWorker.init_calls, 1)
            self.assertEqual(_FakeWorker.shutdown_calls, 1)
            self.assertEqual(_FakeWorker.last_instance.state, "SHUTDOWN")

    def test_ids_unique(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B.", "b.txt": "C. D.", "c.txt": "E."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                run_pipeline(root, out, batch_size=3)
            ids, _ = read_all(out)
            self.assertEqual(len(ids), len(set(ids)))

    def test_batch_size_1_4_16_64(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            self._write_texts(root, {"a.txt": "A. B. C. D. E."})
            for bs in (1, 4, 16, 64):
                out = Path(td) / f"out_{bs}.bin"
                with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                    "ingest_pipeline._split_text", _simple_splitter
                ):
                    res = run_pipeline(root, out, batch_size=bs)
                self.assertEqual(res.records_written, 5)
                self.assertTrue(res.success)

    def test_empty_folder_produces_zero_records(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            root.mkdir()
            out = Path(td) / "out.bin"
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out)
            self.assertEqual(res.files_discovered, 0)
            self.assertEqual(res.records_written, 0)
            self.assertEqual(out.stat().st_size, 0)

    def test_empty_txt_file_yields_zero_sentences(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            root.mkdir()
            (root / "empty.txt").write_bytes(b"")
            out = Path(td) / "out.bin"
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out)
            self.assertEqual(res.total_sentences, 0)
            self.assertEqual(res.records_written, 0)

    def test_non_txt_file_skipped(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            root.mkdir()
            (root / "a.bin").write_bytes(b"\x00\x01")
            out = Path(td) / "out.bin"
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out)
            self.assertEqual(res.files_discovered, 0)
            self.assertEqual(res.files_read, 0)

    def test_whitespace_only_file_yields_zero_sentences(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            root.mkdir()
            (root / "a.txt").write_text("   \n  \n", encoding="utf-8")
            out = Path(td) / "out.bin"
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", lambda text, locale: []
            ):
                res = run_pipeline(root, out)
            self.assertEqual(res.total_sentences, 0)

    def test_summary_counters_present(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B.", "b.txt": "C."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out, batch_size=2)
            self.assertEqual(res.files_discovered, 2)
            self.assertEqual(res.files_read, 2)
            self.assertEqual(res.files_skipped, 0)
            self.assertEqual(res.total_sentences, 3)
            self.assertEqual(res.total_batches, 2)
            self.assertEqual(res.records_written, 3)
            self.assertIsNotNone(res.verification)
            self.assertTrue(res.elapsed_seconds >= 0.0)

    def test_shutdown_and_temp_cleanup_on_encode_error(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            tmp = Path(td) / "out.bin.tmp"
            self._write_texts(root, {"a.txt": "A. B."})
            _FakeWorker.fail_on_encode = True
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out)
            self.assertEqual(_FakeWorker.shutdown_calls, 1)
            self.assertFalse(res.success)
            self.assertTrue(any(e.startswith("[ENCODE]") for e in res.errors))
            self.assertFalse(tmp.exists())
            self.assertFalse(out.exists())

    def test_startup_input_dir_missing(self) -> None:
        self._startup_patcher.stop()
        with TemporaryDirectory() as td:
            root = Path(td) / "missing_dir"
            out = Path(td) / "out.bin"
            res = run_pipeline(root, out)
            self.assertFalse(res.success)
            self.assertTrue(res.errors)
            self.assertIn("[STARTUP]", res.errors[0])
            self.assertEqual(res.records_written, 0)

    def test_startup_output_path_is_directory(self) -> None:
        self._startup_patcher.stop()
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            root.mkdir()
            out_dir = Path(td) / "outdir"
            out_dir.mkdir()
            res = run_pipeline(root, out_dir)
            self.assertFalse(res.success)
            self.assertIn("[STARTUP]", res.errors[0])

    def test_startup_output_parent_missing(self) -> None:
        self._startup_patcher.stop()
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            root.mkdir()
            out = Path(td) / "missing_parent" / "out.bin"
            res = run_pipeline(root, out)
            self.assertFalse(res.success)
            self.assertIn("[STARTUP]", res.errors[0])

    def test_startup_cuda_unavailable(self) -> None:
        self._startup_patcher.stop()
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        with TemporaryDirectory() as td, patch.dict(sys.modules, {"torch": fake_torch}):
            root = Path(td) / "in"
            root.mkdir()
            out = Path(td) / "out.bin"
            res = run_pipeline(root, out)
            self.assertFalse(res.success)
            self.assertTrue(any(e.startswith("[CUDA]") for e in res.errors))

    def test_model_load_failure_returns_result(self) -> None:
        class _FailLoadWorker(_FakeWorker):
            def init(self) -> None:  # type: ignore[override]
                _FakeWorker.init_calls += 1
                raise ModelLoadError("model failed to load")

        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B."})
            with patch("ingest_pipeline.EmbeddingWorker", _FailLoadWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out)
            self.assertFalse(res.success)
            self.assertTrue(any(e.startswith("[MODEL_LOAD]") for e in res.errors))
            self.assertEqual(_FakeWorker.shutdown_calls, 1)
            self.assertFalse(out.exists())

    def test_write_failure_returns_result(self) -> None:
        class _FailWriter:
            def __init__(self, output_path: Path) -> None:
                self.output_path = Path(output_path)
                self.records_written = 0

            def __enter__(self) -> "_FailWriter":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                tmp = self.output_path.with_name(f"{self.output_path.name}.tmp")
                if tmp.exists():
                    tmp.unlink()
                return False

            def write_batch(self, ids: list[int], embeddings: np.ndarray) -> None:
                del ids, embeddings
                raise BinaryWriterError("forced write failure")

        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline.EmbeddingWriter", _FailWriter
            ), patch("ingest_pipeline._split_text", _simple_splitter):
                res = run_pipeline(root, out)
            self.assertFalse(res.success)
            self.assertTrue(any(e.startswith("[WRITE]") for e in res.errors))

    def test_async_write_failure_returns_result(self) -> None:
        class _FailWriter:
            def __init__(self, output_path: Path) -> None:
                self.output_path = Path(output_path)
                self.records_written = 0

            def __enter__(self) -> "_FailWriter":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                tmp = self.output_path.with_name(f"{self.output_path.name}.tmp")
                if tmp.exists():
                    tmp.unlink()
                return False

            def write_batch(self, ids: list[int], embeddings: np.ndarray) -> None:
                del ids, embeddings
                raise BinaryWriterError("async forced write failure")

        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B. C."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline.EmbeddingWriter", _FailWriter
            ), patch("ingest_pipeline._split_text", _simple_splitter):
                res = run_pipeline(root, out, batch_size=2)
            self.assertFalse(res.success)
            self.assertTrue(any(e.startswith("[WRITE]") for e in res.errors))

    def test_overflow_returns_result(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline.batch_sentences_dynamic_cap",
                side_effect=OverflowError("sentence ID 4294967296 exceeds uint32 max"),
            ), patch("ingest_pipeline._split_text", _simple_splitter):
                res = run_pipeline(root, out)
            self.assertFalse(res.success)
            self.assertTrue(any(e.startswith("[OVERFLOW]") for e in res.errors))

    def test_splitter_error_skips_file_continues(self) -> None:
        def _split_with_one_bad(text: str, *, locale: str) -> list[str]:
            del locale
            if "BADFILE" in text:
                raise ValueError("forced splitter failure")
            return _simple_splitter(text, locale="en_US")

        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(
                root,
                {
                    "a.txt": "A. B.",
                    "bad.txt": "BADFILE",
                    "c.txt": "C.",
                },
            )
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _split_with_one_bad
            ):
                res = run_pipeline(root, out, batch_size=4)
            self.assertTrue(res.success)
            self.assertEqual(res.records_written, 3)
            self.assertEqual(res.files_skipped, 1)

    def test_no_output_file_after_model_load_failure(self) -> None:
        class _FailLoadWorker(_FakeWorker):
            def init(self) -> None:  # type: ignore[override]
                raise ModelLoadError("model failed to load")

        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            tmp = Path(td) / "out.bin.tmp"
            self._write_texts(root, {"a.txt": "A."})
            with patch("ingest_pipeline.EmbeddingWorker", _FailLoadWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out)
            self.assertFalse(res.success)
            self.assertFalse(out.exists())
            self.assertFalse(tmp.exists())

    def test_files_skipped_counter_in_result(self) -> None:
        def _split_all_bad(text: str, *, locale: str) -> list[str]:
            del text, locale
            raise ValueError("always bad")

        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A.", "b.txt": "B."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _split_all_bad
            ):
                res = run_pipeline(root, out)
            self.assertTrue(res.success)
            self.assertEqual(res.files_discovered, 2)
            self.assertEqual(res.files_read, 2)
            self.assertEqual(res.files_skipped, 2)
            self.assertEqual(res.records_written, 0)

    @unittest.skipUnless(HAVE_ICU, "PyICU required for real splitter integration test")
    def test_real_splitter_integration(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "First. Second?", "b.txt": "Third!"})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker):
                res = run_pipeline(root, out, batch_size=4, locale="en_US")
            self.assertTrue(res.records_written >= 3)
            self.assertTrue(res.success)

    def test_probe_batch_size_skips_larger_candidates_after_failure(self) -> None:
        worker = _FakeWorker()
        worker.init()
        _FakeWorker.fail_on_batch_size_at_or_above = 64
        picked = probe_batch_size(
            worker,
            ["s"] * 128,
            candidates=(16, 32, 64, 128),
            fallback_batch_size=8,
        )
        self.assertIn(picked, (16, 32))
        self.assertIn(64, _FakeWorker.seen_batch_sizes)
        self.assertNotIn(128, _FakeWorker.seen_batch_sizes)

    def test_probe_max_successful_returns_largest_ok_candidate(self) -> None:
        worker = _FakeWorker()
        worker.init()
        picked = probe_batch_size(
            worker,
            ["s"] * 128,
            candidates=(16, 32, 64, 128),
            fallback_batch_size=8,
            strategy=ProbeStrategy.MAX_SUCCESSFUL,
        )
        self.assertEqual(picked, 128)

    def test_probe_min_latency_prefers_better_per_sentence(self) -> None:
        worker = _FakeWorker()
        worker.init()
        # Two perf_counter calls per candidate: start, end. 64 wins on latency/sentence.
        perf = [
            0.0,
            0.032,
            0.0,
            0.12,
        ]
        with patch("ingest_pipeline.time.perf_counter", side_effect=perf):
            picked = probe_batch_size(
                worker,
                ["s"] * 128,
                candidates=(64, 128),
                fallback_batch_size=8,
                strategy=ProbeStrategy.MIN_LATENCY_PER_SENT,
            )
        self.assertEqual(picked, 64)

    def test_probe_max_successful_returns_128_when_min_latency_would_pick_64(self) -> None:
        worker = _FakeWorker()
        worker.init()
        perf = [
            0.0,
            0.032,
            0.0,
            0.12,
        ]
        with patch("ingest_pipeline.time.perf_counter", side_effect=perf * 2):
            min_pick = probe_batch_size(
                worker,
                ["s"] * 128,
                candidates=(64, 128),
                fallback_batch_size=8,
                strategy=ProbeStrategy.MIN_LATENCY_PER_SENT,
            )
            max_pick = probe_batch_size(
                worker,
                ["s"] * 128,
                candidates=(64, 128),
                fallback_batch_size=8,
                strategy=ProbeStrategy.MAX_SUCCESSFUL,
            )
        self.assertEqual(min_pick, 64)
        self.assertEqual(max_pick, 128)

    def test_probe_epsilon_prefers_largest_within_band(self) -> None:
        worker = _FakeWorker()
        worker.init()
        # 64: 0.032/64 = 0.0005/sent; 128: 0.06656/128 = 0.00052/sent (within 5% of 0.0005)
        perf = [0.0, 0.032, 0.0, 0.06656]
        with patch("ingest_pipeline.time.perf_counter", side_effect=perf):
            picked = probe_batch_size(
                worker,
                ["s"] * 128,
                candidates=(64, 128),
                fallback_batch_size=8,
                strategy=ProbeStrategy.EPSILON,
                probe_epsilon=0.05,
            )
        self.assertEqual(picked, 128)

    def test_resolved_probe_candidates_defaults(self) -> None:
        self.assertEqual(_resolved_probe_candidates(None, None), (16, 32, 64, 128))

    def test_resolved_probe_candidates_max_probe_batch(self) -> None:
        self.assertEqual(_resolved_probe_candidates(None, 64), (16, 32, 64))
        self.assertEqual(
            _resolved_probe_candidates(None, 512),
            (16, 32, 64, 128, 256, 512),
        )
        self.assertEqual(
            _resolved_probe_candidates(None, 1024),
            (16, 32, 64, 128, 256, 512, 1024),
        )

    def test_resolved_probe_candidates_bucketing_includes_1024(self) -> None:
        self.assertEqual(
            _resolved_probe_candidates(None, None, char_len_bucketing=True),
            (16, 32, 64, 128, 256, 512, 1024),
        )

    def test_resolved_probe_candidates_explicit_and_cap(self) -> None:
        self.assertEqual(
            _resolved_probe_candidates((128, 256, 512), 256),
            (128, 256),
        )

    def test_run_pipeline_rejects_empty_probe_candidate_list(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            root.mkdir()
            out = Path(td) / "out.bin"
            with self.assertRaises(ValueError) as ctx:
                run_pipeline(root, out, probe_batch_sizes=())
            self.assertIn("empty", str(ctx.exception).lower())

    def _meta_char_len(self, char_len: int, *, para_line: int = 0) -> SentenceMeta:
        text = "a" * char_len if char_len > 0 else ""
        return SentenceMeta(
            text=text,
            para_line=para_line,
            char_len=char_len,
            doc_para_num=0,
            shard_doc_num=0,
        )

    def test_split_sorted_metas_by_char_len_edges(self) -> None:
        edges = (16, 32)
        metas = sorted(
            [
                self._meta_char_len(5),
                self._meta_char_len(20),
                self._meta_char_len(35),
            ],
            key=lambda m: m.char_len,
        )
        buckets = _split_sorted_metas_by_char_len_edges(metas, edges)
        self.assertEqual(len(buckets), 3)
        self.assertEqual([len(b) for b in buckets], [1, 1, 1])
        self.assertEqual(buckets[0][0].char_len, 5)
        self.assertEqual(buckets[1][0].char_len, 20)
        self.assertEqual(buckets[2][0].char_len, 35)

    def test_split_sorted_metas_empty_returns_empty(self) -> None:
        self.assertEqual(
            _split_sorted_metas_by_char_len_edges([], (16, 32)),
            [],
        )

    def test_validate_char_len_bucket_edges(self) -> None:
        with self.assertRaises(ValueError):
            _validate_char_len_bucket_edges(())
        with self.assertRaises(ValueError):
            _validate_char_len_bucket_edges((32, 16))
        with self.assertRaises(ValueError):
            _validate_char_len_bucket_edges((0, 16))
        with self.assertRaises(ValueError):
            _validate_char_len_bucket_edges((65536,))

    def test_default_char_len_bucket_edges(self) -> None:
        self.assertEqual(
            DEFAULT_CHAR_LEN_BUCKET_EDGES,
            (16, 32, 64, 128, 256, 512, 1024),
        )

    def test_char_len_bucketing_two_buckets_four_probe_calls(self) -> None:
        def _two_band_splitter(text: str, *, locale: str) -> list[str]:
            del text, locale
            shorts = [f"{i:02d}aaaaaaaaaa." for i in range(16)]
            longs = [f"{i:02d}" + "b" * 22 + "." for i in range(16)]
            return shorts + longs

        real_pb = ingest_pipeline_mod.probe_batch_size

        @wraps(real_pb)
        def spy(*a: object, **k: object) -> int:
            spy.probe_count += 1
            return real_pb(*a, **k)

        spy.probe_count = 0

        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            root.mkdir()
            (root / "doc.txt").write_text("x", encoding="utf-8")
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _two_band_splitter
            ), patch("ingest_pipeline.probe_batch_size", spy):
                res = run_pipeline(
                    root,
                    out,
                    batch_size=4,
                    char_len_bucketing=True,
                    max_probe_batch=64,
                )
        self.assertTrue(res.success)
        self.assertEqual(res.records_written, 32)
        self.assertEqual(spy.probe_count, 4)

    def test_per_document_probe_applied_to_each_file(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            doc1 = " ".join(f"D1_{i}." for i in range(20))
            doc2 = " ".join(f"D2_{i}." for i in range(20))
            self._write_texts(root, {"a.txt": doc1, "b.txt": doc2})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ), patch("ingest_pipeline.probe_batch_size", return_value=16) as probe_mock:
                res = run_pipeline(root, out, batch_size=4)
            self.assertTrue(res.success)
            self.assertEqual(res.records_written, 40)
            self.assertEqual(res.total_batches, 4)
            self.assertEqual(probe_mock.call_count, 4)

    def test_file_sentences_sorted_by_char_length_before_encode(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "Long sentence here. Mid. S."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                res = run_pipeline(root, out, batch_size=8)
            self.assertTrue(res.success)
            self.assertTrue(_FakeWorker.seen_batches)
            first_batch = _FakeWorker.seen_batches[0]
            lengths = [len(s) for s in first_batch]
            self.assertEqual(lengths, sorted(lengths))


if __name__ == "__main__":
    unittest.main(verbosity=2)
