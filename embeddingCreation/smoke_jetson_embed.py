#!/usr/bin/env python3
"""
Smoke test: load voyage-4-nano on CUDA with Step 2 settings, run one encode.

Preferred: trust_remote_code=True, attn_implementation=sdpa, torch_dtype=bfloat16,
truncate_dim=2048 (see embeddingCreationPlan.txt Step 2).

Fallbacks (if load fails on your Jetson build — document in logs and try):
  - attn_implementation=\"eager\" or \"sdpa\"
  - torch_dtype=torch.float16 if bfloat16 unsupported
"""

from __future__ import annotations

import argparse
import sys

MODEL_ID = "voyageai/voyage-4-nano"
EXPECTED_DIM = 2048


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentence",
        default="Smoke test sentence for voyage-4-nano.",
        help="text to embed",
    )
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    print("torch:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.version.cuda:", torch.version.cuda)
        print("device_0:", torch.cuda.get_device_name(0))

    model_kwargs: dict = {
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16,
    }
    model = SentenceTransformer(
        MODEL_ID,
        trust_remote_code=True,
        truncate_dim=EXPECTED_DIM,
        model_kwargs=model_kwargs,
    )
    print("SentenceTransformer device:", model.device)

    vec = model.encode(
        args.sentence,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    import numpy as np

    arr = np.asarray(vec)
    if arr.ndim == 2:
        arr = arr[0]
    print("embedding shape:", arr.shape)
    if arr.shape[-1] != EXPECTED_DIM:
        print(f"error: expected dimension {EXPECTED_DIM}", file=sys.stderr)
        return 1
    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
