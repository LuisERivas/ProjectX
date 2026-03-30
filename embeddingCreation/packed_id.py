#!/usr/bin/env python3
"""
Packed 64-bit sentence ID helpers.

Bit layout (LSB -> MSB):
  bits 0..7   : para_line (8)
  bits 8..19  : char_len (12)
  bits 20..33 : doc_para_num (14)
  bits 34..47 : shard_num (14)
  bits 48..63 : shard_doc_num (16)
"""

from __future__ import annotations

from dataclasses import dataclass

PARA_LINE_BITS = 8
CHAR_LEN_BITS = 12
DOC_PARA_BITS = 14
SHARD_BITS = 14
SHARD_DOC_BITS = 16

PARA_LINE_MAX = (1 << PARA_LINE_BITS) - 1
CHAR_LEN_MAX = (1 << CHAR_LEN_BITS) - 1
DOC_PARA_MAX = (1 << DOC_PARA_BITS) - 1
SHARD_MAX = (1 << SHARD_BITS) - 1
SHARD_DOC_MAX = (1 << SHARD_DOC_BITS) - 1
UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class SentenceMeta:
    text: str
    para_line: int
    char_len: int
    doc_para_num: int
    shard_doc_num: int
    shard_num: int = 0


def clamp_char_len(value: int) -> int:
    if value < 0:
        return 0
    if value > CHAR_LEN_MAX:
        return CHAR_LEN_MAX
    return value


def _ensure_range(name: str, value: int, max_value: int) -> None:
    if value < 0 or value > max_value:
        raise ValueError(f"{name} out of range: {value} not in [0, {max_value}]")


def pack_id(
    para_line: int,
    char_len: int,
    doc_para_num: int,
    shard_num: int,
    shard_doc_num: int,
) -> int:
    _ensure_range("para_line", para_line, PARA_LINE_MAX)
    _ensure_range("char_len", char_len, CHAR_LEN_MAX)
    _ensure_range("doc_para_num", doc_para_num, DOC_PARA_MAX)
    _ensure_range("shard_num", shard_num, SHARD_MAX)
    _ensure_range("shard_doc_num", shard_doc_num, SHARD_DOC_MAX)

    packed = (
        (shard_doc_num << 48)
        | (shard_num << 34)
        | (doc_para_num << 20)
        | (char_len << 8)
        | para_line
    )
    if packed < 0 or packed > UINT64_MAX:
        raise ValueError(f"packed id out of uint64 range: {packed}")
    return packed


def unpack_id(packed: int) -> tuple[int, int, int, int, int]:
    _ensure_range("packed", packed, UINT64_MAX)
    para_line = packed & PARA_LINE_MAX
    char_len = (packed >> 8) & CHAR_LEN_MAX
    doc_para_num = (packed >> 20) & DOC_PARA_MAX
    shard_num = (packed >> 34) & SHARD_MAX
    shard_doc_num = (packed >> 48) & SHARD_DOC_MAX
    return para_line, char_len, doc_para_num, shard_num, shard_doc_num
