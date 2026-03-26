#!/usr/bin/env python3
"""
Step 3 test script for embedding worker behavior and guarantees.

Usage:
  python3 test_embedding_worker.py
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from embedding_worker import (
    EXPECTED_DIM,
    EmbeddingError,
    EmbeddingWorker,
    ModelLoadError,
    WorkerState,
    WorkerStateError,
)


class _FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self._available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def _normalized_vec(seed: int) -> np.ndarray:
    vec = np.zeros(EXPECTED_DIM, dtype=np.float32)
    idx = seed % EXPECTED_DIM
    vec[idx] = 1.0
    return vec


def _build_fake_modules(
    *,
    cuda_available: bool = True,
    bfloat16_fails: bool = False,
    force_cpu_device: bool = False,
    encode_oom: bool = False,
) -> tuple[types.ModuleType, types.ModuleType]:
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = _FakeCuda(available=cuda_available)
    torch_mod.bfloat16 = "torch.bfloat16"
    torch_mod.float16 = "torch.float16"

    st_mod = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            if model_id != "voyageai/voyage-4-nano":
                raise RuntimeError("bad model id")
            model_kwargs = kwargs.get("model_kwargs") or {}
            dtype = model_kwargs.get("torch_dtype")
            if bfloat16_fails and dtype == "torch.bfloat16":
                raise RuntimeError("bfloat16 unsupported in test")
            self.device = "cpu" if force_cpu_device else "cuda:0"

        def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
            if encode_oom:
                raise RuntimeError("CUDA out of memory")
            if isinstance(sentences, str):
                return _normalized_vec(len(sentences))
            out = []
            for i, sentence in enumerate(sentences):
                # deterministic and order-sensitive vectors
                out.append(_normalized_vec(i + len(sentence)))
            return np.stack(out, axis=0)

    st_mod.SentenceTransformer = SentenceTransformer
    return torch_mod, st_mod


class TestEmbeddingWorker(unittest.TestCase):
    def test_initial_state(self) -> None:
        w = EmbeddingWorker()
        self.assertEqual(w.stats["state"], WorkerState.UNINITIALIZED.value)

    def test_encode_before_init_raises(self) -> None:
        w = EmbeddingWorker()
        with self.assertRaises(WorkerStateError):
            w.encode_batch(["hello"])

    def test_shutdown_before_init_idempotent(self) -> None:
        w = EmbeddingWorker()
        w.shutdown()
        self.assertEqual(w.stats["state"], WorkerState.SHUTDOWN.value)
        w.shutdown()  # no crash

    def test_init_success_and_double_init_fails(self) -> None:
        torch_mod, st_mod = _build_fake_modules()
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            w.init()
            self.assertEqual(w.stats["state"], WorkerState.READY.value)
            self.assertEqual(w.stats["load_count"], 1)
            self.assertIn("cuda", w.stats["device"] or "")
            with self.assertRaises(WorkerStateError):
                w.init()

    def test_bfloat16_fallback_to_float16(self) -> None:
        torch_mod, st_mod = _build_fake_modules(bfloat16_fails=True)
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            w.init()
            self.assertEqual(w.stats["dtype"], "float16")

    def test_cuda_unavailable_fails_hard(self) -> None:
        torch_mod, st_mod = _build_fake_modules(cuda_available=False)
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            with self.assertRaises(ModelLoadError):
                w.init()

    def test_cpu_device_is_rejected(self) -> None:
        torch_mod, st_mod = _build_fake_modules(force_cpu_device=True)
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            with self.assertRaises(ModelLoadError):
                w.init()

    def test_encode_batch_happy_path_and_order(self) -> None:
        torch_mod, st_mod = _build_fake_modules()
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            w.init()
            sentences = ["alpha", "beta", "gamma", "delta"]
            out = w.encode_batch(sentences)
            self.assertEqual(out.shape, (4, EXPECTED_DIM))
            self.assertEqual(out.dtype, np.float16)

            # Order-sensitive check: different rows should differ by our deterministic mapping.
            self.assertFalse(np.array_equal(out[0], out[1]))
            self.assertFalse(np.array_equal(out[1], out[2]))

            # Post-cast norm sanity from plan.
            norms = np.linalg.norm(out.astype(np.float32), axis=1)
            self.assertTrue(np.all((norms >= 0.95) & (norms <= 1.05)))

    def test_repeated_batches_without_reload(self) -> None:
        torch_mod, st_mod = _build_fake_modules()
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            w.init()
            w.encode_batch(["one"])
            w.encode_batch(["two", "three"])
            stats = w.stats
            self.assertEqual(stats["load_count"], 1)
            self.assertEqual(stats["total_batches"], 2)
            self.assertEqual(stats["total_sentences"], 3)

    def test_malformed_input(self) -> None:
        torch_mod, st_mod = _build_fake_modules()
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            w.init()
            with self.assertRaises(ValueError):
                w.encode_batch([])
            with self.assertRaises(ValueError):
                w.encode_batch("not-a-list")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                w.encode_batch(["ok", 123])  # type: ignore[list-item]

    def test_encode_oom_maps_to_embedding_error(self) -> None:
        torch_mod, st_mod = _build_fake_modules(encode_oom=True)
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            w.init()
            with self.assertRaises(EmbeddingError):
                w.encode_batch(["oom"])

    def test_explicit_shutdown_and_post_shutdown_behavior(self) -> None:
        torch_mod, st_mod = _build_fake_modules()
        with patch.dict(sys.modules, {"torch": torch_mod, "sentence_transformers": st_mod}):
            w = EmbeddingWorker()
            w.init()
            w.shutdown()
            self.assertEqual(w.stats["state"], WorkerState.SHUTDOWN.value)
            with self.assertRaises(WorkerStateError):
                w.encode_batch(["after-shutdown"])
            w.shutdown()  # idempotent


if __name__ == "__main__":
    unittest.main(verbosity=2)

