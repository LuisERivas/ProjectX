#!/usr/bin/env python3
"""
Utilities for parsing Jetson tegrastats output lines.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

_TS_RE = re.compile(r"^(?P<ts>\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
_RAM_RE = re.compile(r"RAM\s+(?P<used>\d+)/(?P<total>\d+)MB")
_SWAP_RE = re.compile(r"SWAP\s+(?P<used>\d+)/(?P<total>\d+)MB")
_CPU_BLOCK_RE = re.compile(r"CPU\s+\[(?P<block>[^\]]+)\]")
_CPU_ENTRY_RE = re.compile(r"(?P<pct>\d+)%@(?P<mhz>\d+)")
_EMC_RE = re.compile(r"EMC_FREQ\s+(?P<pct>\d+)%@(?P<mhz>\d+)")
_GR3D_RE = re.compile(r"GR3D_FREQ\s+(?P<pct>\d+)%@\[(?P<mhz>\d+)\]")
_GR3D_ALT_RE = re.compile(r"GR3D_FREQ\s+(?P<pct>\d+)%@(?P<mhz>\d+)")
_VDD_IN_RE = re.compile(r"VDD_IN\s+(?P<inst>\d+)mW/(?P<avg>\d+)mW")
_TEMP_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)@(?P<temp>\d+(?:\.\d+)?)C")

MAX_CPU_FIELDS = 16


def csv_fieldnames() -> list[str]:
    fields = [
        "timestamp",
        "source",
        "ram_used_mb",
        "ram_total_mb",
        "swap_used_mb",
        "swap_total_mb",
        "cpu_mean_pct",
        "cpu_usages_json",
        "emc_pct",
        "emc_mhz",
        "gr3d_pct",
        "gr3d_mhz",
        "vdd_in_mw",
        "vdd_in_avg_mw",
        "temp_cpu_c",
        "temp_gpu_c",
        "temp_tj_c",
        "temps_json",
        "raw_line",
    ]
    for i in range(MAX_CPU_FIELDS):
        fields.append(f"cpu{i}_pct")
        fields.append(f"cpu{i}_mhz")
    return fields


def _parse_timestamp_iso(line: str) -> str:
    m = _TS_RE.search(line)
    if m is None:
        return datetime.now(timezone.utc).isoformat()
    ts = datetime.strptime(m.group("ts"), "%m-%d-%Y %H:%M:%S")
    return ts.replace(tzinfo=timezone.utc).isoformat()


def parse_tegrastats_line(line: str) -> dict[str, str]:
    row: dict[str, str] = {k: "" for k in csv_fieldnames()}
    row["source"] = "tegrastats"
    row["raw_line"] = line.strip()
    row["timestamp"] = _parse_timestamp_iso(line)

    ram = _RAM_RE.search(line)
    if ram is not None:
        row["ram_used_mb"] = ram.group("used")
        row["ram_total_mb"] = ram.group("total")

    swap = _SWAP_RE.search(line)
    if swap is not None:
        row["swap_used_mb"] = swap.group("used")
        row["swap_total_mb"] = swap.group("total")

    cpu_block = _CPU_BLOCK_RE.search(line)
    cpu_pcts: list[int] = []
    cpu_pairs: list[dict[str, int]] = []
    if cpu_block is not None:
        for i, part in enumerate(cpu_block.group("block").split(",")):
            if i >= MAX_CPU_FIELDS:
                break
            m = _CPU_ENTRY_RE.search(part.strip())
            if m is None:
                continue
            pct = int(m.group("pct"))
            mhz = int(m.group("mhz"))
            cpu_pairs.append({"pct": pct, "mhz": mhz})
            cpu_pcts.append(pct)
            row[f"cpu{i}_pct"] = str(pct)
            row[f"cpu{i}_mhz"] = str(mhz)
    if cpu_pcts:
        row["cpu_mean_pct"] = f"{(sum(cpu_pcts) / len(cpu_pcts)):.4f}"
    row["cpu_usages_json"] = json.dumps(cpu_pairs, separators=(",", ":"))

    emc = _EMC_RE.search(line)
    if emc is not None:
        row["emc_pct"] = emc.group("pct")
        row["emc_mhz"] = emc.group("mhz")

    gr3d = _GR3D_RE.search(line)
    if gr3d is None:
        gr3d = _GR3D_ALT_RE.search(line)
    if gr3d is not None:
        row["gr3d_pct"] = gr3d.group("pct")
        row["gr3d_mhz"] = gr3d.group("mhz")

    vdd = _VDD_IN_RE.search(line)
    if vdd is not None:
        row["vdd_in_mw"] = vdd.group("inst")
        row["vdd_in_avg_mw"] = vdd.group("avg")

    temps: dict[str, float] = {}
    for tm in _TEMP_RE.finditer(line):
        temps[tm.group("name").lower()] = float(tm.group("temp"))
    row["temps_json"] = json.dumps(temps, separators=(",", ":"))
    if "cpu" in temps:
        row["temp_cpu_c"] = f"{temps['cpu']:.4f}"
    if "gpu" in temps:
        row["temp_gpu_c"] = f"{temps['gpu']:.4f}"
    if "tj" in temps:
        row["temp_tj_c"] = f"{temps['tj']:.4f}"

    return row


def psutil_row(
    *,
    timestamp_iso: str,
    ram_used_mb: int,
    ram_total_mb: int,
    swap_used_mb: int,
    swap_total_mb: int,
    cpu_percents: list[float],
) -> dict[str, str]:
    row: dict[str, str] = {k: "" for k in csv_fieldnames()}
    row["timestamp"] = timestamp_iso
    row["source"] = "psutil"
    row["ram_used_mb"] = str(ram_used_mb)
    row["ram_total_mb"] = str(ram_total_mb)
    row["swap_used_mb"] = str(swap_used_mb)
    row["swap_total_mb"] = str(swap_total_mb)
    pairs: list[dict[str, float]] = []
    if cpu_percents:
        row["cpu_mean_pct"] = f"{(sum(cpu_percents) / len(cpu_percents)):.4f}"
    for i, pct in enumerate(cpu_percents[:MAX_CPU_FIELDS]):
        row[f"cpu{i}_pct"] = f"{pct:.4f}"
        pairs.append({"pct": pct})
    row["cpu_usages_json"] = json.dumps(pairs, separators=(",", ":"))
    row["temps_json"] = "{}"
    return row
