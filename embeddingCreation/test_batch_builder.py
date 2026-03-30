#!/usr/bin/env python3
"""Step 6 tests for sentence metadata batching."""

from __future__ import annotations

import unittest

from batch_builder import SentenceBatch, batch_sentences
from packed_id import SentenceMeta


class _TrackedIterable:
    def __init__(self, items: list[SentenceMeta]) -> None:
        self._items = items
        self.consumed = 0

    def __iter__(self):
        for item in self._items:
            self.consumed += 1
            yield item


class TestBatchBuilder(unittest.TestCase):
    def _meta(self, text: str, idx: int) -> SentenceMeta:
        return SentenceMeta(
            text=text,
            para_line=idx,
            char_len=len(text),
            doc_para_num=0,
            shard_doc_num=0,
            shard_num=0,
        )

    def _flatten_sentences(self, batches: list[SentenceBatch]) -> list[str]:
        out: list[str] = []
        for batch in batches:
            out.extend(batch.sentences)
        return out

    def test_batch_size_1(self) -> None:
        s = [self._meta("a", 0), self._meta("b", 1), self._meta("c", 2)]
        batches = list(batch_sentences(s, batch_size=1))
        self.assertEqual(len(batches), 3)
        self.assertEqual([b.sentences for b in batches], [["a"], ["b"], ["c"]])

    def test_batch_size_4_exactly_4(self) -> None:
        s = [self._meta("a", 0), self._meta("b", 1), self._meta("c", 2), self._meta("d", 3)]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].sentences, ["a", "b", "c", "d"])

    def test_batch_size_4_with_7_sentences(self) -> None:
        s = [self._meta(f"s{i}", i) for i in range(1, 8)]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].sentences, ["s1", "s2", "s3", "s4"])
        self.assertEqual(batches[1].sentences, ["s5", "s6", "s7"])

    def test_batch_size_4_with_3_sentences(self) -> None:
        s = [self._meta("x", 0), self._meta("y", 1), self._meta("z", 2)]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].sentences, ["x", "y", "z"])

    def test_batch_size_larger_than_total(self) -> None:
        s = [self._meta(v, i) for i, v in enumerate(["a", "b", "c", "d", "e"])]
        batches = list(batch_sentences(s, batch_size=100))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].sentences, ["a", "b", "c", "d", "e"])

    def test_zero_sentences_yields_no_batches(self) -> None:
        self.assertEqual(list(batch_sentences([], batch_size=4)), [])

    def test_order_preservation(self) -> None:
        s = [self._meta(f"sentence-{i}", i) for i in range(13)]
        batches = list(batch_sentences(s, batch_size=4))
        self.assertEqual(self._flatten_sentences(batches), [f"sentence-{i}" for i in range(13)])

    def test_no_dropped_or_duplicated_sentences(self) -> None:
        s = [self._meta(f"s-{i}", i) for i in range(23)]
        batches = list(batch_sentences(s, batch_size=8))
        flattened = self._flatten_sentences(batches)
        expected = [f"s-{i}" for i in range(23)]
        self.assertEqual(len(flattened), len(expected))
        self.assertEqual(flattened, expected)

    def test_multiple_files_span_batch_boundaries(self) -> None:
        files = [
            ["f1s1", "f1s2", "f1s3"],
            ["f2s1", "f2s2", "f2s3"],
        ]

        def stream():
            for group in files:
                for idx, sentence in enumerate(group):
                    yield self._meta(sentence, idx)

        batches = list(batch_sentences(stream(), batch_size=4))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].sentences, ["f1s1", "f1s2", "f1s3", "f2s1"])
        self.assertEqual(batches[1].sentences, ["f2s2", "f2s3"])

    def test_batch_size_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences([self._meta("a", 0)], batch_size=0))

    def test_batch_size_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences([self._meta("a", 0)], batch_size=-1))

    def test_char_budget_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences([self._meta("a", 0)], batch_size=4, char_budget=0))

    def test_char_budget_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences([self._meta("a", 0)], batch_size=4, char_budget=-3))

    def test_char_budget_forces_early_flush(self) -> None:
        s = [
            self._meta("aa", 0),      # char_len=2
            self._meta("bbbb", 1),    # char_len=4
            self._meta("cccccc", 2),  # char_len=6
            self._meta("ddddd", 3),   # char_len=5
        ]
        batches = list(batch_sentences(s, batch_size=4, char_budget=10))
        self.assertEqual([b.sentences for b in batches], [["aa", "bbbb"], ["cccccc"], ["ddddd"]])

    def test_char_budget_still_respects_batch_size_cap(self) -> None:
        s = [self._meta("a", i) for i in range(7)]
        batches = list(batch_sentences(s, batch_size=3, char_budget=999))
        self.assertEqual([len(b.sentences) for b in batches], [3, 3, 1])

    def test_char_budget_allows_single_long_sentence(self) -> None:
        s = [self._meta("x" * 20, 0), self._meta("y", 1)]
        batches = list(batch_sentences(s, batch_size=4, char_budget=8))
        self.assertEqual([b.sentences for b in batches], [["x" * 20], ["y"]])

    def test_deterministic_same_input_same_output(self) -> None:
        s = [self._meta(f"s{i}", i) for i in range(7)]
        run1 = list(batch_sentences(s, batch_size=3))
        run2 = list(batch_sentences(s, batch_size=3))
        self.assertEqual(len(run1), len(run2))
        for b1, b2 in zip(run1, run2):
            self.assertEqual(b1.sentences, b2.sentences)

    def test_generator_laziness(self) -> None:
        tracked = _TrackedIterable(
            [self._meta("a", 0), self._meta("b", 1), self._meta("c", 2), self._meta("d", 3), self._meta("e", 4)]
        )
        gen = batch_sentences(tracked, batch_size=2)
        first = next(gen)
        self.assertEqual(first.sentences, ["a", "b"])
        self.assertEqual(tracked.consumed, 2)

    def test_sentence_batch_fields_same_length(self) -> None:
        s = [self._meta(f"s{i}", i) for i in range(9)]
        batches = list(batch_sentences(s, batch_size=4))
        for batch in batches:
            self.assertEqual(len(batch.metas), len(batch.sentences))
            self.assertGreater(len(batch.metas), 0)

    def test_non_meta_item_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(batch_sentences([self._meta("ok", 0), "bad"]))  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main(verbosity=2)
