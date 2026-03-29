#!/usr/bin/env python3
"""
Step 6 tests for sentence batching and ID mapping.

Usage:
  python3 test_batch_builder.py
"""

from __future__ import annotations

import unittest

from batch_builder import UINT32_MAX, SentenceBatch, batch_sentences


class _TrackedIterable:
    def __init__(self, items: list[str]) -> None:
        self._items = items
        self.consumed = 0

    def __iter__(self):
        for item in self._items:
            self.consumed += 1
            yield item


class TestBatchBuilder(unittest.TestCase):
    def _flatten_ids(self, batches: list[SentenceBatch]) -> list[int]:
        out: list[int] = []
        for batch in batches:
            out.extend(batch.ids)
        return out

    def _flatten_sentences(self, batches: list[SentenceBatch]) -> list[str]:
        out: list[str] = []
        for batch in batches:
            out.extend(batch.sentences)
        return out

    def test_batch_size_1(self) -> None:
        s = ["a", "b", "c"]
        batches = list(batch_sentences(s, batch_size=1))
        self.assertEqual(len(batches), 3)
        self.assertEqual([b.ids for b in batches], [[0], [1], [2]])
        self.assertEqual([b.sentences for b in batches], [["a"], ["b"], ["c"]])

    def test_batch_size_4_exactly_4(self) -> None:
        s = ["a", "b", "c", "d"]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].ids, [0, 1, 2, 3])
        self.assertEqual(batches[0].sentences, s)

    def test_batch_size_4_with_7_sentences(self) -> None:
        s = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].ids, [0, 1, 2, 3])
        self.assertEqual(batches[1].ids, [4, 5, 6])
        self.assertEqual(batches[0].sentences, s[:4])
        self.assertEqual(batches[1].sentences, s[4:])

    def test_batch_size_4_with_3_sentences(self) -> None:
        s = ["x", "y", "z"]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].ids, [0, 1, 2])
        self.assertEqual(batches[0].sentences, s)

    def test_batch_size_larger_than_total(self) -> None:
        s = ["a", "b", "c", "d", "e"]
        batches = list(batch_sentences(s, batch_size=100))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].ids, [0, 1, 2, 3, 4])

    def test_zero_sentences_yields_no_batches(self) -> None:
        self.assertEqual(list(batch_sentences([], batch_size=4)), [])

    def test_start_id_offset(self) -> None:
        s = ["a", "b", "c"]
        batches = list(batch_sentences(s, batch_size=10, start_id=10))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].ids, [10, 11, 12])

    def test_id_continuity_across_batches(self) -> None:
        s = [f"s{i}" for i in range(11)]
        batches = list(batch_sentences(s, batch_size=4))
        all_ids = self._flatten_ids(batches)
        self.assertEqual(all_ids, list(range(11)))

    def test_order_preservation(self) -> None:
        s = [f"sentence-{i}" for i in range(13)]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(self._flatten_sentences(batches), s)

    def test_no_dropped_or_duplicated_sentences(self) -> None:
        s = [f"s-{i}" for i in range(23)]
        batches = list(batch_sentences(s, batch_size=8))
        flattened = self._flatten_sentences(batches)
        self.assertEqual(len(flattened), len(s))
        self.assertEqual(flattened, s)

    def test_multiple_files_span_batch_boundaries(self) -> None:
        files = [
            ["f1s1", "f1s2", "f1s3"],
            ["f2s1", "f2s2", "f2s3"],
        ]

        def stream():
            for group in files:
                for sentence in group:
                    yield sentence

        batches = list(batch_sentences(stream(), batch_size=4))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].sentences, ["f1s1", "f1s2", "f1s3", "f2s1"])
        self.assertEqual(batches[1].sentences, ["f2s2", "f2s3"])
        self.assertEqual(self._flatten_ids(batches), list(range(6)))

    def test_batch_size_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences(["a"], batch_size=0))

    def test_batch_size_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences(["a"], batch_size=-1))

    def test_start_id_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences(["a"], start_id=-1))

    def test_overflow_raises(self) -> None:
        s = ["a", "b", "c"]
        with self.assertRaises(OverflowError):
            list(batch_sentences(s, batch_size=2, start_id=UINT32_MAX - 1))

    def test_exact_uint32_max_boundary_succeeds(self) -> None:
        batches = list(batch_sentences(["last"], batch_size=1, start_id=UINT32_MAX))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].ids, [UINT32_MAX])
        self.assertEqual(batches[0].sentences, ["last"])

    def test_exact_uint32_max_plus_one_overflows(self) -> None:
        with self.assertRaises(OverflowError):
            list(batch_sentences(["ok", "boom"], batch_size=4, start_id=UINT32_MAX))

    def test_deterministic_same_input_same_output(self) -> None:
        s = [f"s{i}" for i in range(7)]
        run1 = list(batch_sentences(s, batch_size=3))
        run2 = list(batch_sentences(s, batch_size=3))
        self.assertEqual(len(run1), len(run2))
        for b1, b2 in zip(run1, run2):
            self.assertEqual(b1.ids, b2.ids)
            self.assertEqual(b1.sentences, b2.sentences)

    def test_generator_laziness(self) -> None:
        tracked = _TrackedIterable(["a", "b", "c", "d", "e"])
        gen = batch_sentences(tracked, batch_size=2)
        first = next(gen)
        self.assertEqual(first.sentences, ["a", "b"])
        self.assertEqual(tracked.consumed, 2)

    def test_sentence_batch_fields_same_length(self) -> None:
        s = [f"s{i}" for i in range(9)]
        batches = list(batch_sentences(s, batch_size=4))
        for batch in batches:
            self.assertEqual(len(batch.ids), len(batch.sentences))
            self.assertGreater(len(batch.ids), 0)

    def test_non_str_sentence_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences(["ok", 123]))  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main(verbosity=2)
