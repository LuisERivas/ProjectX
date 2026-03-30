#!/usr/bin/env python3
from __future__ import annotations

import unittest

from packed_id import (
    CHAR_LEN_MAX,
    DOC_PARA_MAX,
    PARA_LINE_MAX,
    SHARD_DOC_MAX,
    SHARD_MAX,
    SentenceMeta,
    clamp_char_len,
    pack_id,
    unpack_id,
)


class TestPackedId(unittest.TestCase):
    def test_round_trip(self) -> None:
        values = (12, 301, 22, 9, 100)
        packed = pack_id(*values)
        self.assertEqual(unpack_id(packed), values)

    def test_max_values_round_trip(self) -> None:
        values = (PARA_LINE_MAX, CHAR_LEN_MAX, DOC_PARA_MAX, SHARD_MAX, SHARD_DOC_MAX)
        packed = pack_id(*values)
        self.assertEqual(unpack_id(packed), values)

    def test_overflow_raises(self) -> None:
        with self.assertRaises(ValueError):
            pack_id(PARA_LINE_MAX + 1, 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            pack_id(1, CHAR_LEN_MAX + 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            pack_id(1, 1, DOC_PARA_MAX + 1, 1, 1)
        with self.assertRaises(ValueError):
            pack_id(1, 1, 1, SHARD_MAX + 1, 1)
        with self.assertRaises(ValueError):
            pack_id(1, 1, 1, 1, SHARD_DOC_MAX + 1)

    def test_uniqueness(self) -> None:
        a = pack_id(1, 100, 2, 0, 10)
        b = pack_id(1, 101, 2, 0, 10)
        self.assertNotEqual(a, b)
        self.assertEqual(a, pack_id(1, 100, 2, 0, 10))

    def test_clamp_char_len(self) -> None:
        self.assertEqual(clamp_char_len(-2), 0)
        self.assertEqual(clamp_char_len(100), 100)
        self.assertEqual(clamp_char_len(999999), CHAR_LEN_MAX)

    def test_sentence_meta_defaults(self) -> None:
        meta = SentenceMeta("hello", para_line=0, char_len=5, doc_para_num=0, shard_doc_num=1)
        self.assertEqual(meta.shard_num, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
