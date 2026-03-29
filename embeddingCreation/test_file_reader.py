#!/usr/bin/env python3
"""
Step 5 tests for text file discovery and reading.

Usage:
  python3 test_file_reader.py
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from file_reader import discover_files, read_text_files

try:
    from sentence_splitter import split_sentences
except ModuleNotFoundError:
    split_sentences = None


class TestFileReader(unittest.TestCase):
    def test_multiple_valid_txt_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b.txt").write_text("B", encoding="utf-8")
            (root / "a.txt").write_text("A", encoding="utf-8")
            (root / "c.txt").write_text("C", encoding="utf-8")

            paths = discover_files(root)
            self.assertEqual([p.name for p in paths], ["a.txt", "b.txt", "c.txt"])

            rows = list(read_text_files(root))
            self.assertEqual([(p.name, t) for p, t in rows], [("a.txt", "A"), ("b.txt", "B"), ("c.txt", "C")])

    def test_empty_folder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(discover_files(root), [])
            self.assertEqual(list(read_text_files(root)), [])

    def test_zero_byte_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "good.txt").write_text("hello", encoding="utf-8")
            (root / "empty.txt").write_bytes(b"")
            rows = list(read_text_files(root))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0].name, "good.txt")

    def test_non_text_extensions_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "doc.txt").write_text("ok", encoding="utf-8")
            (root / "img.png").write_bytes(b"\x89PNG")
            (root / "blob.bin").write_bytes(b"\x00\x01")
            paths = discover_files(root)
            self.assertEqual([p.name for p in paths], ["doc.txt"])

    def test_invalid_utf8_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "good.txt").write_text("good", encoding="utf-8")
            (root / "bad.txt").write_bytes(b"\xff\xfe")
            rows = list(read_text_files(root))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0].name, "good.txt")

    def test_recursive_false_ignores_nested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "top.txt").write_text("top", encoding="utf-8")
            sub = root / "nested"
            sub.mkdir()
            (sub / "inner.txt").write_text("inner", encoding="utf-8")

            paths = discover_files(root, recursive=False)
            self.assertEqual([p.name for p in paths], ["top.txt"])

    def test_recursive_true_includes_nested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "top.txt").write_text("top", encoding="utf-8")
            sub = root / "nested"
            sub.mkdir()
            (sub / "inner.txt").write_text("inner", encoding="utf-8")

            paths = discover_files(root, recursive=True)
            self.assertEqual([p.name for p in paths], ["inner.txt", "top.txt"])

    def test_hidden_files_skipped_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".hidden.txt").write_text("hidden", encoding="utf-8")
            (root / "visible.txt").write_text("visible", encoding="utf-8")

            paths = discover_files(root)
            self.assertEqual([p.name for p in paths], ["visible.txt"])

    def test_hidden_files_included_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".hidden.txt").write_text("hidden", encoding="utf-8")
            (root / "visible.txt").write_text("visible", encoding="utf-8")

            paths = discover_files(root, skip_hidden=False)
            self.assertEqual([p.name for p in paths], [".hidden.txt", "visible.txt"])

    def test_symlink_file_supported(self) -> None:
        if os.name == "nt" or not hasattr(os, "symlink"):
            self.skipTest("symlink test requires Unix-like environment")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target.txt"
            link = root / "link.txt"
            target.write_text("linked", encoding="utf-8")
            os.symlink(target, link)

            rows = list(read_text_files(root))
            names = sorted([p.name for p, _ in rows])
            self.assertEqual(names, ["link.txt", "target.txt"])

    def test_large_file_over_limit_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "small.txt").write_text("ok", encoding="utf-8")
            (root / "big.txt").write_text("x" * 32, encoding="utf-8")

            paths = discover_files(root, max_file_size_bytes=8)
            self.assertEqual([p.name for p in paths], ["small.txt"])

    def test_mixed_extensions_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.md").write_text("b", encoding="utf-8")
            (root / "c.log").write_text("c", encoding="utf-8")

            paths = discover_files(root, extensions=frozenset({".txt", ".md"}))
            self.assertEqual([p.name for p in paths], ["a.txt", "b.md"])

    def test_extension_matching_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "A.TXT").write_text("A", encoding="utf-8")
            (root / "B.TxT").write_text("B", encoding="utf-8")

            paths = discover_files(root)
            self.assertEqual([p.name for p in paths], ["A.TXT", "B.TxT"])

    def test_deterministic_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ["z.txt", "m.txt", "a.txt"]:
                (root / name).write_text(name, encoding="utf-8")
            first = [p.name for p in discover_files(root)]
            second = [p.name for p in discover_files(root)]
            self.assertEqual(first, second)
            self.assertEqual(first, ["a.txt", "m.txt", "z.txt"])

    def test_directory_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(FileNotFoundError):
                discover_files(root / "missing")

            file_path = root / "not_a_dir.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                discover_files(file_path)

    def test_custom_encoding_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            text = "olá"
            (root / "latin1.txt").write_bytes(text.encode("latin-1"))

            rows = list(read_text_files(root, encoding="latin-1"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], text)

    def test_integration_with_sentence_splitter(self) -> None:
        if split_sentences is None:
            self.skipTest("sentence_splitter/PyICU not available in this environment")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "doc.txt").write_text("First sentence. Second sentence.", encoding="utf-8")
            rows = list(read_text_files(root))
            self.assertEqual(len(rows), 1)
            _, text = rows[0]
            sentences = split_sentences(text)
            self.assertEqual(sentences, ["First sentence.", "Second sentence."])


if __name__ == "__main__":
    unittest.main(verbosity=2)
