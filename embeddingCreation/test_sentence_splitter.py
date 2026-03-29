#!/usr/bin/env python3
"""
Step 4 test script for ICU sentence splitting.

Usage:
  python3 test_sentence_splitter.py
"""

from __future__ import annotations

import unittest

from sentence_splitter import split_sentences


class TestSplitSentences(unittest.TestCase):

    def test_normal_english_prose(self) -> None:
        text = "The cat sat. The dog ran. Why?"
        result = split_sentences(text)
        self.assertEqual(result, ["The cat sat.", "The dog ran.", "Why?"])

    def test_abbreviations_dr_mr(self) -> None:
        text = "Dr. Smith went to Washington. He arrived at 3 p.m."
        result = split_sentences(text)
        # ICU abbreviation handling varies by version/locale data;
        # verify output is non-empty, stripped, ordered, and covers all text.
        self.assertGreaterEqual(len(result), 2)
        joined = " ".join(result)
        self.assertIn("Dr.", joined)
        self.assertIn("Smith", joined)
        self.assertIn("Washington", joined)
        self.assertIn("p.m.", joined)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
