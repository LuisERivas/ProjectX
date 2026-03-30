#!/usr/bin/env python3
from __future__ import annotations

import unittest

from paragraph_splitter import split_into_paragraphs


class TestParagraphSplitter(unittest.TestCase):
    def test_single_newline_splits(self) -> None:
        text = "para1\npara2\npara3"
        self.assertEqual(split_into_paragraphs(text), ["para1", "para2", "para3"])

    def test_empty_segments_dropped(self) -> None:
        text = "a\n\n\nb\n"
        self.assertEqual(split_into_paragraphs(text), ["a", "b"])

    def test_whitespace_segments_dropped(self) -> None:
        text = "a\n   \n\t\nb"
        self.assertEqual(split_into_paragraphs(text), ["a", "b"])

    def test_order_preserved(self) -> None:
        text = "first\nsecond\nthird"
        self.assertEqual(split_into_paragraphs(text), ["first", "second", "third"])

    def test_non_str_raises(self) -> None:
        with self.assertRaises(ValueError):
            split_into_paragraphs(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
