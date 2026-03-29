#!/usr/bin/env python3
"""
Step 9 tests for binary writer format and safety guarantees.

Usage:
  python3 test_binary_writer.py
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from binary_writer import (
    EMBEDDING_BYTES,
    EXPECTED_DIM,
    RECORD_SIZE,
    UINT32_MAX,
    BinaryWriterError,
    EmbeddingWriter,
)


def _unit_embedding(dim_index: int) -> np.ndarray:
    arr = np.zeros((1, EXPECTED_DIM), dtype=np.float16)
    arr[0, dim_index] = np.float16(1.0)
    return arr


def _unit_batch(count: int) -> np.ndarray:
    arr = np.zeros((count, EXPECTED_DIM), dtype=np.float16)
    for i in range(count):
        arr[i, i % EXPECTED_DIM] = np.float16(1.0)
    return arr


def _read_records(path: Path) -> tuple[list[int], np.ndarray]:
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    with path.open("rb") as fp:
        raw = fp.read()
    if len(raw) % RECORD_SIZE != 0:
        raise AssertionError("invalid file size alignment")
    n = len(raw) // RECORD_SIZE
    for i in range(n):
        base = i * RECORD_SIZE
        rid = struct.unpack("<I", raw[base : base + 4])[0]
        emb_bytes = raw[base + 4 : base + RECORD_SIZE]
        emb = np.frombuffer(emb_bytes, dtype="<f2")
        vectors.append(emb.astype(np.float16))
        ids.append(rid)
    if not vectors:
        return ids, np.zeros((0, EXPECTED_DIM), dtype=np.float16)
    return ids, np.stack(vectors, axis=0)


class TestBinaryWriter(unittest.TestCase):
    def test_write_zero_records_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out):
                pass
            self.assertTrue(out.exists())
            self.assertEqual(out.stat().st_size, 0)

    def test_write_one_record_size_4100(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([0], _unit_embedding(0))
            self.assertEqual(out.stat().st_size, 4100)

    def test_write_three_records_size_12300(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([0, 1, 2], _unit_batch(3))
            self.assertEqual(out.stat().st_size, 12300)

    def test_hex_record0_matches_appendix_b(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([0], _unit_embedding(0))
            with out.open("rb") as fp:
                b = fp.read(8)
            self.assertEqual(b, b"\x00\x00\x00\x00\x00\x3c\x00\x00")

    def test_hex_record1_offset_4100_matches_appendix_b(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            emb = np.zeros((2, EXPECTED_DIM), dtype=np.float16)
            emb[0, 0] = np.float16(1.0)
            emb[1, 1] = np.float16(1.0)
            with EmbeddingWriter(out) as w:
                w.write_batch([0, 1], emb)
            with out.open("rb") as fp:
                fp.seek(4100)
                b = fp.read(8)
            self.assertEqual(b, b"\x01\x00\x00\x00\x00\x00\x00\x3c")

    def test_round_trip_ids_and_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            ids = [5, 6, 7]
            emb = _unit_batch(3)
            with EmbeddingWriter(out) as w:
                w.write_batch(ids, emb)
            got_ids, got_emb = _read_records(out)
            self.assertEqual(got_ids, ids)
            self.assertTrue(np.array_equal(got_emb, emb))

    def test_uint32_is_little_endian(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([1], _unit_embedding(0))
            with out.open("rb") as fp:
                self.assertEqual(fp.read(4), b"\x01\x00\x00\x00")

    def test_float16_component_is_little_endian(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([0], _unit_embedding(0))
            with out.open("rb") as fp:
                fp.seek(4)
                self.assertEqual(fp.read(2), b"\x00\x3c")

    def test_id_embedding_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                with self.assertRaises(BinaryWriterError):
                    w.write_batch([0, 1], _unit_batch(3))

    def test_wrong_dtype_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            bad = _unit_batch(1).astype(np.float32)
            with EmbeddingWriter(out) as w:
                with self.assertRaises(BinaryWriterError):
                    w.write_batch([0], bad)

    def test_wrong_dimension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            bad = np.zeros((1, 1024), dtype=np.float16)
            with EmbeddingWriter(out) as w:
                with self.assertRaises(BinaryWriterError):
                    w.write_batch([0], bad)

    def test_large_batch_100_records_size(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch(list(range(100)), _unit_batch(100))
            self.assertEqual(out.stat().st_size, 410000)

    def test_multiple_write_batch_calls_accumulate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([0, 1], _unit_batch(2))
                w.write_batch([2, 3], _unit_batch(2))
                w.write_batch([4, 5], _unit_batch(2))
                self.assertEqual(w.records_written, 6)
            self.assertEqual(out.stat().st_size, 24600)

    def test_file_valid_after_close_size_modulo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([0, 1, 2], _unit_batch(3))
            self.assertEqual(out.stat().st_size % RECORD_SIZE, 0)

    def test_temp_cleanup_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            tmp = Path(td) / "emb.bin.tmp"
            with self.assertRaises(RuntimeError):
                with EmbeddingWriter(out) as w:
                    w.write_batch([0], _unit_embedding(0))
                    raise RuntimeError("forced")
            self.assertFalse(out.exists())
            self.assertFalse(tmp.exists())

    def test_write_after_close_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            w = EmbeddingWriter(out)
            w.close()
            with self.assertRaises(BinaryWriterError):
                w.write_batch([0], _unit_embedding(0))

    def test_context_manager_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                w.write_batch([0], _unit_embedding(0))
            self.assertTrue(out.exists())

    def test_id_out_of_uint32_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                with self.assertRaises(BinaryWriterError):
                    w.write_batch([UINT32_MAX + 1], _unit_embedding(0))

    def test_negative_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "emb.bin"
            with EmbeddingWriter(out) as w:
                with self.assertRaises(BinaryWriterError):
                    w.write_batch([-1], _unit_embedding(0))

    def test_record_payload_bytes_count(self) -> None:
        row = _unit_embedding(0)[0]
        self.assertEqual(len(row.astype("<f2").tobytes()), EMBEDDING_BYTES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
