#!/usr/bin/env python3
"""
Step 11 tests for end-to-end pipeline orchestration.

Usage:
  python3 test_ingest_pipeline.py
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from binary_reader import read_all
from ingest_pipeline import run_pipeline

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
        if _FakeWorker.fail_on_encode:
            raise RuntimeError("forced encode failure")
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
            self.assertEqual(out.stat().st_size, 5 * 4100)

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

    def test_ids_globally_monotonic_and_unique(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            out = Path(td) / "out.bin"
            self._write_texts(root, {"a.txt": "A. B.", "b.txt": "C. D.", "c.txt": "E."})
            with patch("ingest_pipeline.EmbeddingWorker", _FakeWorker), patch(
                "ingest_pipeline._split_text", _simple_splitter
            ):
                run_pipeline(root, out, batch_size=3)
            ids, _ = read_all(out)
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(len(ids), len(set(ids)))

    def test_batch_size_1_4_16(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td) / "in"
            self._write_texts(root, {"a.txt": "A. B. C. D. E."})
            for bs in (1, 4, 16):
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
                with self.assertRaises(RuntimeError):
                    run_pipeline(root, out)
            self.assertEqual(_FakeWorker.shutdown_calls, 1)
            self.assertFalse(tmp.exists())
            self.assertFalse(out.exists())

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
