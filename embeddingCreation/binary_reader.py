#!/usr/bin/env python3
"""
Step 10 binary reader/verifier for embedding records.

Reads records in the format:
  uint32 id (little-endian) + float16[2048] embedding (little-endian)
"""

from __future__ import annotations

import logging
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

RECORD_SIZE: int = 4100
EXPECTED_DIM: int = 2048
ID_BYTES: int = 4
EMBEDDING_BYTES: int = EXPECTED_DIM * 2

LOGGER = logging.getLogger("binary_reader")


class BinaryReaderError(Exception):
    """Raised when binary reader cannot parse file safely."""


@dataclass(frozen=True)
class VerificationReport:
    path: str
    file_size: int
    record_count: int
    id_min: int | None
    id_max: int | None
    ids_monotonic: bool
    ids_unique: bool
    norms_min: float | None
    norms_max: float | None
    norms_mean: float | None
    norms_all_in_tolerance: bool
    all_finite: bool
    errors: list[str]
    ok: bool


def _as_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise BinaryReaderError(f"file does not exist: {p}")
    if p.is_dir():
        raise BinaryReaderError(f"path is a directory, expected file: {p}")
    return p


def record_count(path: str | Path) -> int:
    p = _as_path(path)
    size = p.stat().st_size
    if size % RECORD_SIZE != 0:
        raise BinaryReaderError(
            f"invalid file size alignment: {size} bytes is not divisible by {RECORD_SIZE}"
        )
    return size // RECORD_SIZE


def iter_records(path: str | Path) -> Generator[tuple[int, np.ndarray], None, None]:
    p = _as_path(path)
    _ = record_count(p)  # validates size alignment
    try:
        with p.open("rb") as fp:
            while True:
                block = fp.read(RECORD_SIZE)
                if not block:
                    return
                if len(block) != RECORD_SIZE:
                    raise BinaryReaderError(
                        f"truncated record encountered: expected {RECORD_SIZE} bytes, got {len(block)}"
                    )
                rid = struct.unpack("<I", block[:ID_BYTES])[0]
                emb = np.frombuffer(block[ID_BYTES:], dtype="<f2").astype(np.float16)
                if emb.shape[0] != EXPECTED_DIM:
                    raise BinaryReaderError(
                        f"decoded embedding dim mismatch: got {emb.shape[0]}, expected {EXPECTED_DIM}"
                    )
                yield rid, emb
    except OSError as exc:
        raise BinaryReaderError(f"failed reading file: {p}") from exc


def read_all(path: str | Path) -> tuple[list[int], np.ndarray]:
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    for rid, emb in iter_records(path):
        ids.append(rid)
        vectors.append(emb)
    if not vectors:
        return ids, np.zeros((0, EXPECTED_DIM), dtype=np.float16)
    return ids, np.stack(vectors, axis=0).astype(np.float16, copy=False)


def read_record(path: str | Path, index: int) -> tuple[int, np.ndarray]:
    if index < 0:
        raise BinaryReaderError(f"record index must be >= 0, got {index}")

    p = _as_path(path)
    total = record_count(p)
    if index >= total:
        raise BinaryReaderError(f"record index out of bounds: {index} >= {total}")

    offset = index * RECORD_SIZE
    try:
        with p.open("rb") as fp:
            fp.seek(offset)
            block = fp.read(RECORD_SIZE)
    except OSError as exc:
        raise BinaryReaderError(f"failed random-access read for {p}") from exc

    if len(block) != RECORD_SIZE:
        raise BinaryReaderError(
            f"short read at index {index}: expected {RECORD_SIZE}, got {len(block)}"
        )
    rid = struct.unpack("<I", block[:ID_BYTES])[0]
    emb = np.frombuffer(block[ID_BYTES:], dtype="<f2").astype(np.float16)
    if emb.shape[0] != EXPECTED_DIM:
        raise BinaryReaderError(
            f"decoded embedding dim mismatch: got {emb.shape[0]}, expected {EXPECTED_DIM}"
        )
    return rid, emb


def verify_file(
    path: str | Path,
    *,
    norm_min: float = 0.95,
    norm_max: float = 1.05,
) -> VerificationReport:
    p = _as_path(path)
    size = p.stat().st_size
    count = record_count(p)
    ids, embeddings = read_all(p)

    errors: list[str] = []
    ids_monotonic = True
    ids_unique = True
    all_finite = True
    norms_all_in_tolerance = True

    if len(ids) != count:
        errors.append(f"record count mismatch: read ids={len(ids)} expected={count}")

    for i in range(len(ids) - 1):
        if not (ids[i] < ids[i + 1]):
            ids_monotonic = False
            errors.append(f"ids not strictly increasing at index {i}: {ids[i]} -> {ids[i + 1]}")
            break

    if len(set(ids)) != len(ids):
        ids_unique = False
        errors.append("duplicate ids detected")

    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_DIM:
        errors.append(
            f"embedding shape mismatch: got {embeddings.shape}, expected (N, {EXPECTED_DIM})"
        )

    emb32 = embeddings.astype(np.float32, copy=False)
    if not np.isfinite(emb32).all():
        all_finite = False
        errors.append("non-finite embedding values detected (NaN or Inf)")

    norms_min: float | None = None
    norms_max: float | None = None
    norms_mean: float | None = None
    if emb32.shape[0] > 0:
        norms = np.linalg.norm(emb32, axis=1)
        norms_min = float(norms.min())
        norms_max = float(norms.max())
        norms_mean = float(norms.mean())
        in_range = (norms >= norm_min) & (norms <= norm_max)
        finite_norms = np.isfinite(norms)
        bad_mask = (~in_range) | (~finite_norms)
        if np.any(bad_mask):
            norms_all_in_tolerance = False
            bad = np.where(bad_mask)[0]
            first_bad = int(bad[0])
            errors.append(
                f"norm out of tolerance at record {first_bad}: {float(norms[first_bad]):.6f} not in [{norm_min}, {norm_max}]"
            )

    ok = len(errors) == 0
    report = VerificationReport(
        path=str(p),
        file_size=size,
        record_count=count,
        id_min=min(ids) if ids else None,
        id_max=max(ids) if ids else None,
        ids_monotonic=ids_monotonic,
        ids_unique=ids_unique,
        norms_min=norms_min,
        norms_max=norms_max,
        norms_mean=norms_mean,
        norms_all_in_tolerance=norms_all_in_tolerance,
        all_finite=all_finite,
        errors=errors,
        ok=ok,
    )
    if ok:
        LOGGER.info(
            "binary verify ok: records=%d size=%d path=%s",
            report.record_count,
            report.file_size,
            report.path,
        )
    else:
        LOGGER.error(
            "binary verify failed: records=%d errors=%d path=%s",
            report.record_count,
            len(report.errors),
            report.path,
        )
        for err in report.errors:
            LOGGER.error("verify issue: %s", err)
    return report


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python3 binary_reader.py <file.bin>", file=sys.stderr)
        return 1
    try:
        report = verify_file(argv[1])
    except BinaryReaderError as exc:
        print(f"verify_error: {exc}", file=sys.stderr)
        return 2

    print(f"path: {report.path}")
    print(f"file_size: {report.file_size}")
    print(f"record_count: {report.record_count}")
    print(f"id_min: {report.id_min}")
    print(f"id_max: {report.id_max}")
    print(f"ids_monotonic: {report.ids_monotonic}")
    print(f"ids_unique: {report.ids_unique}")
    print(f"all_finite: {report.all_finite}")
    print(f"norms_min: {report.norms_min}")
    print(f"norms_max: {report.norms_max}")
    print(f"norms_mean: {report.norms_mean}")
    print(f"norms_all_in_tolerance: {report.norms_all_in_tolerance}")
    print(f"ok: {report.ok}")
    if report.errors:
        print("errors:")
        for err in report.errors:
            print(f"- {err}")
    return 0 if report.ok else 3


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
