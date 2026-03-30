#!/usr/bin/env python3
"""
Step 4 test script for ICU sentence splitting.

Usage:
  python3 test_sentence_splitter.py
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import sentence_splitter as sentence_splitter_mod
    from sentence_splitter import _reset_break_iterator_cache_for_tests, split_sentences

    HAVE_ICU = True
except Exception:
    sentence_splitter_mod = None
    HAVE_ICU = False

    def split_sentences(text: str, *, locale: str = "en_US") -> list[str]:
        del text, locale
        raise RuntimeError("PyICU not installed")

    def _reset_break_iterator_cache_for_tests() -> None:
        return None


@unittest.skipUnless(HAVE_ICU, "PyICU required for sentence splitter tests")
class TestSplitSentences(unittest.TestCase):

    def test_normal_english_prose(self) -> None:
        text = "The cat sat. The dog ran. Why?"
        result = split_sentences(text)
        self.assertEqual(result, ["The cat sat.", "The dog ran.", "Why?"])

    def test_abbreviations_dr_mr(self) -> None:
        text = "Dr. Smith went to Washington. He arrived at 3 p.m."
        result = split_sentences(text)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].startswith("Dr. Smith"))
        self.assertIn("p.m.", result[1])

    def test_abbreviations_mrs_prof(self) -> None:
        text = "Mrs. Jones called Prof. Adams. They met at 9 a.m."
        result = split_sentences(text)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].startswith("Mrs. Jones"))

    def test_abbreviation_eg_ie(self) -> None:
        text = "Use a tool, e.g. a hammer. It works, i.e. it is effective."
        result = split_sentences(text)
        self.assertEqual(len(result), 2)

    def test_exclamation_question_combo(self) -> None:
        text = "Really?! That's amazing!"
        result = split_sentences(text)
        self.assertEqual(len(result), 2)

    def test_multiline_with_blank_lines(self) -> None:
        text = "First sentence.\n\nSecond sentence.\n"
        result = split_sentences(text)
        self.assertEqual(result, ["First sentence.", "Second sentence."])

    def test_whitespace_only_input(self) -> None:
        self.assertEqual(split_sentences("   \n\t  "), [])

    def test_empty_string(self) -> None:
        self.assertEqual(split_sentences(""), [])

    def test_very_short_fragment(self) -> None:
        self.assertEqual(split_sentences("Yes."), ["Yes."])

    def test_no_terminal_punctuation(self) -> None:
        text = "This text has no ending period"
        result = split_sentences(text)
        self.assertEqual(result, ["This text has no ending period"])

    def test_ellipsis(self) -> None:
        text = "Wait... What happened?"
        result = split_sentences(text)
        self.assertTrue(len(result) >= 1)
        self.assertIn("What happened?", result[-1])

    def test_non_str_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            split_sentences(123)  # type: ignore[arg-type]

    def test_crlf_normalization(self) -> None:
        text = "Line one.\r\nLine two.\r"
        result = split_sentences(text)
        self.assertEqual(result, ["Line one.", "Line two."])

    def test_single_character(self) -> None:
        self.assertEqual(split_sentences("A"), ["A"])

    def test_order_preservation(self) -> None:
        text = "First. Second. Third. Fourth. Fifth."
        result = split_sentences(text)
        self.assertEqual(
            result,
            ["First.", "Second.", "Third.", "Fourth.", "Fifth."],
        )

    def test_all_results_stripped_and_nonempty(self) -> None:
        text = "  Hello.   \n   World.  "
        result = split_sentences(text)
        for s in result:
            self.assertEqual(s, s.strip())
            self.assertTrue(len(s) > 0)

    def test_multiple_spaces_between_sentences(self) -> None:
        text = "One.     Two.     Three."
        result = split_sentences(text)
        self.assertEqual(result, ["One.", "Two.", "Three."])

    def test_locale_parameter_accepted(self) -> None:
        text = "Hello. World."
        result = split_sentences(text, locale="root")
        self.assertEqual(len(result), 2)

    def test_break_iterator_reuse_same_locale(self) -> None:
        _reset_break_iterator_cache_for_tests()
        with patch(
            "sentence_splitter.icu.BreakIterator.createSentenceInstance",
            wraps=sentence_splitter_mod.icu.BreakIterator.createSentenceInstance,
        ) as spy:
            split_sentences("One. Two.", locale="en_US")
            split_sentences("Three. Four.", locale="en_US")
            self.assertEqual(spy.call_count, 1)

    def test_break_iterator_new_instance_for_different_locale(self) -> None:
        _reset_break_iterator_cache_for_tests()
        with patch(
            "sentence_splitter.icu.BreakIterator.createSentenceInstance",
            wraps=sentence_splitter_mod.icu.BreakIterator.createSentenceInstance,
        ) as spy:
            split_sentences("One. Two.", locale="en_US")
            split_sentences("Un. Deux.", locale="fr_FR")
            self.assertEqual(spy.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
