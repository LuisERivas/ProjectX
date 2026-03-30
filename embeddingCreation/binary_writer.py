#!/usr/bin/env python3
"""
Step 9 binary writer for embedding records.

Writes records as:
  uint64 id (little-endian) + float16[2048] embedding (little-endian)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

RECORD_SIZE: int = 4104
EXPECTED_DIM: int = 2048
EMBEDDING_BYTES: int = EXPECTED_DIM * 2  # float16 => 2 bytes each
UINT64_MAX: int = (2**64) - 1

LOGGER = logging.getLogger("binary_writer")


class BinaryWriterError(Exception):
    """Raised when binary writer validation or I/O fails."""


class EmbeddingWriter:
    """Context-managed writer for ProjectX embedding binary records."""

    def __init__(self, output_path: str | Path) -> None:
        self._output_path = Path(output_path)
        self._tmp_path = self._output_path.with_name(f"{self._output_path.name}.tmp")
        self._records = 0
        self._closed = False
        self._fp = None

        parent = self._output_path.parent
        if not parent.exists():
            raise BinaryWriterError(f"output directory does not exist: {parent}")
        if parent.is_file():
            raise BinaryWriterError(f"output parent is not a directory: {parent}")

        try:
            self._fp = self._tmp_path.open("wb")
        except OSError as exc:
            raise BinaryWriterError(f"failed to open temp output file: {self._tmp_path}") from exc

    @property
    def records_written(self) -> int:
        return self._records

    @property
    def output_path(self) -> Path:
        return self._output_path

    def write_batch(self, ids: list[int], embeddings: np.ndarray) -> None:
        if self._closed or self._fp is None:
            raise BinaryWriterError("writer is already closed")

        arr = np.asarray(embeddings)
        if arr.dtype != np.float16:
            raise BinaryWriterError(f"expected float16 embeddings, got {arr.dtype}")
        if arr.ndim != 2:
            raise BinaryWriterError(f"expected embeddings ndim=2, got ndim={arr.ndim}")
        if arr.shape[1] != EXPECTED_DIM:
            raise BinaryWriterError(f"expected embedding dim {EXPECTED_DIM}, got {arr.shape[1]}")
        if len(ids) != arr.shape[0]:
            raise BinaryWriterError(
                f"id count ({len(ids)}) != embedding count ({arr.shape[0]})"
            )

        try:
            id_arr = np.asarray(ids, dtype=np.int64)
        except Exception as exc:
            raise BinaryWriterError("ids must be int-convertible values") from exc
        if id_arr.ndim != 1 or id_arr.shape[0] != len(ids):
            raise BinaryWriterError("ids must be a flat list of integers")
        if np.any(id_arr < 0) or np.any(id_arr > UINT64_MAX):
            bad = int(id_arr[(id_arr < 0) | (id_arr > UINT64_MAX)][0])
            raise BinaryWriterError(f"id {bad} outside uint64 range")

        le_arr = arr.astype("<f2", copy=False)
        record_dtype = np.dtype([("id", "<u8"), ("emb", ("<f2", EXPECTED_DIM))])
        block = np.empty(len(ids), dtype=record_dtype)
        block["id"] = id_arr.astype("<u8", copy=False)
        block["emb"] = le_arr
        payload = block.tobytes()
        if len(payload) != len(ids) * RECORD_SIZE:
            raise BinaryWriterError(
                f"record block byte size mismatch: got {len(payload)}, expected {len(ids) * RECORD_SIZE}"
            )

        try:
            self._fp.write(payload)
        except OSError as exc:
            raise BinaryWriterError(f"write failure for temp file: {self._tmp_path}") from exc

        self._records += len(ids)
        LOGGER.info("wrote records: batch=%d total=%d", len(ids), self._records)

    def close(self) -> None:
        if self._closed:
            return
        if self._fp is None:
            self._closed = True
            return

        try:
            self._fp.flush()
            os.fsync(self._fp.fileno())
            self._fp.close()
            self._fp = None
            self._tmp_path.replace(self._output_path)

            file_size = self._output_path.stat().st_size
            expected = self._records * RECORD_SIZE
            if file_size != expected:
                raise BinaryWriterError(
                    f"file size mismatch after close: got {file_size}, expected {expected}"
                )
            if file_size % RECORD_SIZE != 0:
                raise BinaryWriterError(
                    f"invalid file size alignment after close: {file_size} % {RECORD_SIZE} != 0"
                )

            LOGGER.info(
                "binary write complete: records=%d bytes=%d path=%s",
                self._records,
                file_size,
                self._output_path,
            )
            self._closed = True
        except OSError as exc:
            raise BinaryWriterError(f"failed to finalize binary output: {self._output_path}") from exc

    def _cleanup_temp(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None
        try:
            if self._tmp_path.exists():
                self._tmp_path.unlink()
        except OSError:
            pass
        self._closed = True

    def __enter__(self) -> EmbeddingWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
            return False
        self._cleanup_temp()
        LOGGER.error("binary writer aborted; temp file cleaned: %s", self._tmp_path)
        return False
