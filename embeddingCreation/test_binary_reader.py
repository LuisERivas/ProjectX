#!/usr/bin/env python3
"""
Step 10 tests for binary reader/verifier utility.

Usage:
  python3 test_binary_reader.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from binary_reader import (
    BinaryReaderError,
    EXPECTED_DIM,
    RECORD_SIZE,
    iter_records,
    read_all,
    read_record,
    record_count,
    verify_file,
)
from binary_writer import EmbeddingWriter


def _unit_batch(count: int) -> np.ndarray:
    arr = np.zeros((count, EXPECTED_DIM), dtype=np.float16)
    for i in range(count):
        arr[i, i % EXPECTED_DIM] = np.float16(1.0)
    return arr


def _manual_write(path: Path, ids: list[int], embeddings: np.ndarray) -> None:
    with path.open("wb") as fp:
        for i, rid in enumerate(ids):
            fp.write(int(rid).to_bytes(10, byteorder="little", signed=False))
            fp.write(embeddings[i].astype("<f2", copy=False).tobytes())


class TestBinaryReader(unittest.TestCase):
    def test_read_back_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "one.bin"
            ids = [0]
            emb = _unit_batch(1)
            with EmbeddingWriter(path) as w:
                w.write_batch(ids, emb)
            got_ids, got_emb = read_all(path)
            self.assertEqual(got_ids, ids)
            self.assertTrue(np.array_equal(got_emb, emb))

    def test_read_back_three_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "three.bin"
            ids = [10, 11, 12]
            emb = _unit_batch(3)
            with EmbeddingWriter(path) as w:
                w.write_batch(ids, emb)
            got_ids, got_emb = read_all(path)
            self.assertEqual(got_ids, ids)
            self.assertTrue(np.array_equal(got_emb, emb))

    def test_read_back_hundred_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hundred.bin"
            ids = list(range(100))
            emb = _unit_batch(100)
            with EmbeddingWriter(path) as w:
                w.write_batch(ids, emb)
            got_ids, got_emb = read_all(path)
            self.assertEqual(len(got_ids), 100)
            self.assertEqual(got_emb.shape, (100, EXPECTED_DIM))
            self.assertEqual(got_emb.dtype, np.float16)

    def test_read_empty_file_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.bin"
            with EmbeddingWriter(path):
                pass
            got_ids, got_emb = read_all(path)
            self.assertEqual(got_ids, [])
            self.assertEqual(got_emb.shape, (0, EXPECTED_DIM))

    def test_unaligned_file_size_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.bin"
            path.write_bytes(b"\x00" * (RECORD_SIZE + 1))
            with self.assertRaises(BinaryReaderError):
                read_all(path)

    def test_record_count_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "count.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([0, 1, 2, 3, 4], _unit_batch(5))
            self.assertEqual(record_count(path), 5)

    def test_verify_first_and_last_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ids.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch(list(range(10)), _unit_batch(10))
            report = verify_file(path)
            self.assertEqual(report.id_min, 0)
            self.assertEqual(report.id_max, 9)
            self.assertTrue(report.ok)

    def test_verify_id_monotonicity_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mono_ok.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([2, 3, 4], _unit_batch(3))
            report = verify_file(path)
            self.assertTrue(report.ids_monotonic)
            self.assertTrue(report.ok)

    def test_verify_id_monotonicity_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mono_bad.bin"
            ids = [0, 2, 1]
            emb = _unit_batch(3)
            _manual_write(path, ids, emb)
            report = verify_file(path)
            self.assertFalse(report.ids_monotonic)
            self.assertTrue(report.ok)

    def test_verify_id_uniqueness_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "uniq_ok.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([4, 5, 6], _unit_batch(3))
            report = verify_file(path)
            self.assertTrue(report.ids_unique)

    def test_verify_id_uniqueness_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "uniq_bad.bin"
            ids = [7, 7, 8]
            emb = _unit_batch(3)
            _manual_write(path, ids, emb)
            report = verify_file(path)
            self.assertFalse(report.ids_unique)
            self.assertFalse(report.ok)

    def test_verify_embedding_dimension_2048(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dim.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([0, 1], _unit_batch(2))
            _, emb = read_all(path)
            self.assertEqual(emb.shape[1], EXPECTED_DIM)

    def test_verify_norms_near_unit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "norm_ok.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([0, 1, 2], _unit_batch(3))
            report = verify_file(path)
            self.assertTrue(report.norms_all_in_tolerance)
            self.assertTrue(report.ok)

    def test_detect_nan_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nan.bin"
            emb = _unit_batch(2)
            emb[1, 5] = np.float16(np.nan)
            _manual_write(path, [0, 1], emb)
            report = verify_file(path)
            self.assertFalse(report.all_finite)
            self.assertFalse(report.ok)

    def test_detect_inf_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "inf.bin"
            emb = _unit_batch(2)
            emb[1, 5] = np.float16(np.inf)
            _manual_write(path, [0, 1], emb)
            report = verify_file(path)
            self.assertFalse(report.all_finite)
            self.assertFalse(report.ok)

    def test_round_trip_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "roundtrip.bin"
            ids = list(range(8))
            emb = _unit_batch(8)
            with EmbeddingWriter(path) as w:
                w.write_batch(ids, emb)
            got_ids, got_emb = read_all(path)
            self.assertEqual(got_ids, ids)
            self.assertTrue(np.array_equal(got_emb, emb))

    def test_random_access_read_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "random.bin"
            ids = [10, 11, 12, 13, 14]
            emb = _unit_batch(5)
            with EmbeddingWriter(path) as w:
                w.write_batch(ids, emb)
            rid, vec = read_record(path, 2)
            self.assertEqual(rid, 12)
            self.assertTrue(np.array_equal(vec, emb[2]))

    def test_random_access_out_of_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "oob.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([0, 1, 2, 3, 4], _unit_batch(5))
            with self.assertRaises(BinaryReaderError):
                read_record(path, 999)

    def test_iter_records_yields_all(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "iter.bin"
            ids = [0, 1, 2, 3]
            emb = _unit_batch(4)
            with EmbeddingWriter(path) as w:
                w.write_batch(ids, emb)
            out_ids: list[int] = []
            out_vecs: list[np.ndarray] = []
            for rid, vec in iter_records(path):
                out_ids.append(rid)
                out_vecs.append(vec)
            self.assertEqual(out_ids, ids)
            self.assertEqual(len(out_vecs), 4)
            self.assertTrue(np.array_equal(np.stack(out_vecs, axis=0), emb))

    def test_verify_valid_file_ok_true(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "verify_ok.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([0, 1, 2], _unit_batch(3))
            report = verify_file(path)
            self.assertTrue(report.ok)
            self.assertEqual(report.record_count, 3)

    def test_verify_empty_file_ok_true(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "verify_empty.bin"
            with EmbeddingWriter(path):
                pass
            report = verify_file(path)
            self.assertTrue(report.ok)
            self.assertEqual(report.record_count, 0)

    def test_cli_summary_output_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cli.bin"
            with EmbeddingWriter(path) as w:
                w.write_batch([0], _unit_batch(1))

            script = Path(__file__).with_name("binary_reader.py")
            proc = subprocess.run(
                [sys.executable, str(script), str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("record_count: 1", proc.stdout)
            self.assertIn("ok: True", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
