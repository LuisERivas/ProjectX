#!/usr/bin/env python3
"""
Step 7 tests for embedding validation and conversion.

Usage:
  python3 test_embedding_validator.py
"""

from __future__ import annotations

import unittest

import numpy as np

from embedding_validator import EmbeddingValidationError, validate_embeddings

EXPECTED_DIM = 2048


def _unit_batch(n: int) -> np.ndarray:
    arr = np.zeros((n, EXPECTED_DIM), dtype=np.float32)
    for i in range(n):
        arr[i, i % EXPECTED_DIM] = 1.0
    return arr


class TestEmbeddingValidator(unittest.TestCase):
    def test_single_embedding_happy_path(self) -> None:
        arr = _unit_batch(1)
        out = validate_embeddings(arr, expected_count=1)
        self.assertEqual(out.shape, (1, EXPECTED_DIM))
        self.assertEqual(out.dtype, np.float16)

    def test_batch_happy_path(self) -> None:
        arr = _unit_batch(4)
        out = validate_embeddings(arr, expected_count=4)
        self.assertEqual(out.shape, (4, EXPECTED_DIM))
        self.assertEqual(out.dtype, np.float16)

    def test_count_mismatch_fewer_rows(self) -> None:
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(_unit_batch(2), expected_count=3)

    def test_count_mismatch_more_rows(self) -> None:
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(_unit_batch(4), expected_count=3)

    def test_wrong_dimension(self) -> None:
        arr = np.zeros((3, 1024), dtype=np.float32)
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(arr, expected_count=3, expected_dim=EXPECTED_DIM)

    def test_non_finite_nan_in_fp32(self) -> None:
        arr = _unit_batch(1)
        arr[0, 0] = np.nan
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(arr, expected_count=1)

    def test_non_finite_pos_inf_in_fp32(self) -> None:
        arr = _unit_batch(1)
        arr[0, 0] = np.inf
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(arr, expected_count=1)

    def test_non_finite_neg_inf_in_fp32(self) -> None:
        arr = _unit_batch(1)
        arr[0, 0] = -np.inf
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(arr, expected_count=1)

    def test_norm_exactly_one_fp32(self) -> None:
        arr = _unit_batch(2)
        out = validate_embeddings(arr, expected_count=2)
        norms = np.linalg.norm(out.astype(np.float32), axis=1)
        self.assertTrue(np.all((norms >= 0.95) & (norms <= 1.05)))

    def test_norm_well_outside_before_renorm_passes(self) -> None:
        arr = _unit_batch(1) * 2.0
        out = validate_embeddings(arr, expected_count=1)
        self.assertEqual(out.dtype, np.float16)

    def test_zero_vector_raises(self) -> None:
        arr = np.zeros((1, EXPECTED_DIM), dtype=np.float32)
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(arr, expected_count=1)

    def test_post_cast_tolerance_happy_path(self) -> None:
        arr = _unit_batch(3)
        out = validate_embeddings(arr, expected_count=3)
        post_norms = np.linalg.norm(out.astype(np.float32), axis=1)
        self.assertTrue(np.all((post_norms >= 0.95) & (post_norms <= 1.05)))

    def test_dtype_assertion_float16(self) -> None:
        arr = _unit_batch(2)
        out = validate_embeddings(arr, expected_count=2)
        self.assertEqual(out.dtype, np.float16)

    def test_return_shape_matches_input(self) -> None:
        arr = _unit_batch(5)
        out = validate_embeddings(arr, expected_count=5)
        self.assertEqual(out.shape, (5, EXPECTED_DIM))

    def test_all_same_value_vector_passes(self) -> None:
        arr = np.ones((1, EXPECTED_DIM), dtype=np.float32)
        out = validate_embeddings(arr, expected_count=1)
        self.assertEqual(out.shape, (1, EXPECTED_DIM))

    def test_custom_tolerance_enforced(self) -> None:
        arr = _unit_batch(1)
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(
                arr,
                expected_count=1,
                pre_cast_norm_min=0.999,
                pre_cast_norm_max=1.001,
                post_cast_norm_min=1.1,
                post_cast_norm_max=1.2,
            )

    def test_post_cast_finiteness_for_normal_input(self) -> None:
        arr = _unit_batch(1)
        out = validate_embeddings(arr, expected_count=1)
        self.assertTrue(np.isfinite(out.astype(np.float32)).all())

    def test_accepts_1d_input_as_single_embedding(self) -> None:
        arr = np.zeros((EXPECTED_DIM,), dtype=np.float32)
        arr[0] = 1.0
        out = validate_embeddings(arr, expected_count=1)
        self.assertEqual(out.shape, (1, EXPECTED_DIM))

    def test_skip_norm_checks_happy_path(self) -> None:
        arr = _unit_batch(2) * 3.0
        out = validate_embeddings(arr, expected_count=2, skip_norm_checks=True)
        self.assertEqual(out.shape, (2, EXPECTED_DIM))
        self.assertEqual(out.dtype, np.float16)

    def test_skip_norm_checks_still_enforces_shape(self) -> None:
        bad = np.zeros((2, 1024), dtype=np.float32)
        with self.assertRaises(EmbeddingValidationError):
            validate_embeddings(bad, expected_count=2, skip_norm_checks=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
