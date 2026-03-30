#!/usr/bin/env python3
"""
Run `run_ingest.py` with background resource monitoring.

This launcher is Jetson-first:
- On Jetson with `tegrastats` available, it records full metrics.
- On non-Jetson systems, pass `--fallback-psutil` to record RAM/swap/CPU only.

Outputs are written to `run_ingest_data/` by default:
- `metrics.csv`
- `run_metadata.json`
- `summary.png` (if matplotlib is available)
- `summary.html` (optional via `--html`)

README snippet:
    cd embeddingCreation
    python3 run_ingest_monitored.py --output-dir run_ingest_data --interval-ms 1000 -- \
      --input inputText --output outputText/out.bin

If `tegrastats` requires privilege on your image, try:
    sudo tegrastats --interval 1000
or run this launcher with sufficient permissions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tegrastats_parse import csv_fieldnames, parse_tegrastats_line, psutil_row

DEFAULT_INTERVAL_MS = 1000
DEFAULT_OUTPUT_DIR = "run_ingest_data"
TEGRA_RELEASE = Path("/etc/nv_tegra_release")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_jetson_host() -> bool:
    machine = platform.machine().lower()
    if machine not in ("aarch64", "arm64"):
        return False
    return platform.system() == "Linux" and TEGRA_RELEASE.is_file()


def _float_or_none(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _write_summary_plot(csv_path: Path, png_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"[monitor] matplotlib unavailable; skipping summary.png ({exc})",
            file=sys.stderr,
        )
        return False

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        print("[monitor] no metrics rows; skipping summary.png", file=sys.stderr)
        return False

    x: list[float] = []
    first_ts: datetime | None = None
    for idx, row in enumerate(rows):
        ts_raw = row.get("timestamp") or ""
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            x.append(float(idx))
            continue
        if first_ts is None:
            first_ts = ts
        x.append((ts - first_ts).total_seconds())

    ram_used = [_float_or_none(r.get("ram_used_mb", "")) for r in rows]
    ram_total = [_float_or_none(r.get("ram_total_mb", "")) for r in rows]
    swap_used = [_float_or_none(r.get("swap_used_mb", "")) for r in rows]
    cpu_mean = [_float_or_none(r.get("cpu_mean_pct", "")) for r in rows]
    emc_pct = [_float_or_none(r.get("emc_pct", "")) for r in rows]
    gr3d_pct = [_float_or_none(r.get("gr3d_pct", "")) for r in rows]
    emc_mhz = [_float_or_none(r.get("emc_mhz", "")) for r in rows]
    gr3d_mhz = [_float_or_none(r.get("gr3d_mhz", "")) for r in rows]
    vdd_inst = [_float_or_none(r.get("vdd_in_mw", "")) for r in rows]
    vdd_avg = [_float_or_none(r.get("vdd_in_avg_mw", "")) for r in rows]
    temp_cpu = [_float_or_none(r.get("temp_cpu_c", "")) for r in rows]
    temp_gpu = [_float_or_none(r.get("temp_gpu_c", "")) for r in rows]
    temp_tj = [_float_or_none(r.get("temp_tj_c", "")) for r in rows]

    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    ax = axes.flatten()

    def _plot_series(axis: Any, series: list[float | None], label: str) -> bool:
        y = [v for v in series if v is not None]
        if not y:
            return False
        axis.plot(x, series, label=label, linewidth=1.4)
        return True

    if _plot_series(ax[0], ram_used, "RAM used (MB)"):
        _plot_series(ax[0], ram_total, "RAM total (MB)")
        ax[0].set_title("RAM")
        ax[0].legend(loc="upper left")
    else:
        ax[0].text(0.1, 0.5, "No RAM data", transform=ax[0].transAxes)

    if _plot_series(ax[1], swap_used, "Swap used (MB)"):
        ax[1].set_title("Swap")
        ax[1].legend(loc="upper left")
    else:
        ax[1].text(0.1, 0.5, "No swap data", transform=ax[1].transAxes)

    if _plot_series(ax[2], cpu_mean, "CPU mean (%)"):
        ax[2].set_title("CPU")
        ax[2].set_ylim(0, 100)
        ax[2].legend(loc="upper left")
    else:
        ax[2].text(0.1, 0.5, "No CPU data", transform=ax[2].transAxes)

    had_util = False
    if _plot_series(ax[3], emc_pct, "EMC (%)"):
        had_util = True
    if _plot_series(ax[3], gr3d_pct, "GR3D (%)"):
        had_util = True
    if had_util:
        ax[3].set_title("EMC / GR3D Utilization")
        ax[3].set_ylim(0, 100)
        ax[3].legend(loc="upper left")
    else:
        ax[3].text(0.1, 0.5, "No EMC/GR3D data", transform=ax[3].transAxes)

    had_power = False
    if _plot_series(ax[4], vdd_inst, "VDD_IN inst (mW)"):
        had_power = True
    if _plot_series(ax[4], vdd_avg, "VDD_IN avg (mW)"):
        had_power = True
    if had_power:
        ax[4].set_title("Power")
        ax[4].legend(loc="upper left")
    else:
        ax[4].text(0.1, 0.5, "No VDD_IN data", transform=ax[4].transAxes)

    had_temp = False
    if _plot_series(ax[5], temp_cpu, "CPU temp (C)"):
        had_temp = True
    if _plot_series(ax[5], temp_gpu, "GPU temp (C)"):
        had_temp = True
    if _plot_series(ax[5], temp_tj, "TJ temp (C)"):
        had_temp = True
    if had_temp:
        ax[5].set_title("Temperatures")
        ax[5].legend(loc="upper left")
    else:
        ax[5].text(0.1, 0.5, "No temperature data", transform=ax[5].transAxes)

    had_clocks = False
    if _plot_series(ax[6], emc_mhz, "EMC clock (MHz)"):
        had_clocks = True
    if _plot_series(ax[6], gr3d_mhz, "GR3D clock (MHz)"):
        had_clocks = True
    if had_clocks:
        ax[6].set_title("Clocks")
        ax[6].legend(loc="upper left")
    else:
        ax[6].text(0.1, 0.5, "No clock data", transform=ax[6].transAxes)

    sources = [r.get("source", "") for r in rows]
    if any(sources):
        source_nums = [1.0 if s == "tegrastats" else 0.0 for s in sources]
        ax[7].plot(x, source_nums, linewidth=1.2)
        ax[7].set_yticks([0.0, 1.0], labels=["psutil", "tegrastats"])
        ax[7].set_title("Metric Source")
    else:
        ax[7].text(0.1, 0.5, "No source data", transform=ax[7].transAxes)

    for a in ax:
        a.grid(alpha=0.3)
        a.set_xlabel("Elapsed seconds")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return True


def _write_summary_html(png_path: Path, html_path: Path) -> None:
    png_bytes = png_path.read_bytes()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>run_ingest monitor summary</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;}img{max-width:100%;"
        "border:1px solid #ddd;}</style></head><body>"
        "<h1>run_ingest monitor summary</h1>"
        "<p>Generated by run_ingest_monitored.py</p>"
        f"<img src='data:image/png;base64,{b64}' alt='summary chart' />"
        "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")


def _start_tegrastats(
    *,
    tegrastats_path: str,
    interval_ms: int,
    writer: csv.DictWriter,
    csv_file: io.TextIOBase,
    stop_event: threading.Event,
    sample_counter: list[int],
) -> tuple[subprocess.Popen[str], threading.Thread]:
    proc = subprocess.Popen(
        [tegrastats_path, "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if stop_event.is_set():
                break
            row = parse_tegrastats_line(line)
            writer.writerow(row)
            csv_file.flush()
            sample_counter[0] += 1

    thread = threading.Thread(target=_reader, name="tegrastats-reader", daemon=True)
    thread.start()
    return proc, thread


def _start_psutil_sampler(
    *,
    interval_ms: int,
    writer: csv.DictWriter,
    csv_file: io.TextIOBase,
    stop_event: threading.Event,
    sample_counter: list[int],
) -> threading.Thread:
    try:
        import psutil  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "psutil is required for --fallback-psutil. Install with: pip install psutil"
        ) from exc

    psutil.cpu_percent(interval=None, percpu=True)

    def _sampler() -> None:
        while not stop_event.is_set():
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            cpu_per = list(psutil.cpu_percent(interval=None, percpu=True))
            row = psutil_row(
                timestamp_iso=_now_iso(),
                ram_used_mb=int(vm.used // (1024 * 1024)),
                ram_total_mb=int(vm.total // (1024 * 1024)),
                swap_used_mb=int(sm.used // (1024 * 1024)),
                swap_total_mb=int(sm.total // (1024 * 1024)),
                cpu_percents=[float(v) for v in cpu_per],
            )
            writer.writerow(row)
            csv_file.flush()
            sample_counter[0] += 1
            stop_event.wait(interval_ms / 1000.0)

    thread = threading.Thread(target=_sampler, name="psutil-sampler", daemon=True)
    thread.start()
    return thread


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for metrics.csv, run_metadata.json, summary.png",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help=f"Sampling interval in milliseconds (default: {DEFAULT_INTERVAL_MS})",
    )
    parser.add_argument(
        "--tegrastats-path",
        default="tegrastats",
        help="Path to tegrastats binary/command (default: tegrastats)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create run_ingest_data.zip archive after run",
    )
    parser.add_argument(
        "--fallback-psutil",
        action="store_true",
        help="Use psutil RAM/swap/CPU sampling when tegrastats is unavailable",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write summary.html with embedded summary.png",
    )
    parser.add_argument(
        "ingest_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to run_ingest.py. Put them after --.",
    )
    args = parser.parse_args(argv)
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be > 0")
    ingest_args = list(args.ingest_args)
    if ingest_args and ingest_args[0] == "--":
        ingest_args = ingest_args[1:]
    if not ingest_args:
        parser.error("No run_ingest.py args provided. Example: -- --input in --output out.bin")
    args.ingest_args = ingest_args
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "metrics.csv"
    metadata_json = output_dir / "run_metadata.json"
    summary_png = output_dir / "summary.png"
    summary_html = output_dir / "summary.html"

    start_time = _now_iso()
    is_jetson = _is_jetson_host()
    tegrastats_resolved = shutil.which(args.tegrastats_path)

    monitor_mode = ""
    stop_event = threading.Event()
    sample_counter = [0]
    monitor_proc: subprocess.Popen[str] | None = None
    monitor_thread: threading.Thread | None = None
    monitor_stderr = ""

    with metrics_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames(), extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()

        if is_jetson and tegrastats_resolved is not None:
            monitor_mode = "tegrastats"
            monitor_proc, monitor_thread = _start_tegrastats(
                tegrastats_path=tegrastats_resolved,
                interval_ms=args.interval_ms,
                writer=writer,
                csv_file=csv_file,
                stop_event=stop_event,
                sample_counter=sample_counter,
            )
            time.sleep(0.2)
            if monitor_proc.poll() is not None:
                err = ""
                if monitor_proc.stderr is not None:
                    err = monitor_proc.stderr.read().strip()
                msg = (
                    f"tegrastats exited early (code {monitor_proc.returncode}). "
                    f"Path: {tegrastats_resolved}. stderr: {err}"
                )
                if args.fallback_psutil:
                    print(f"[monitor] {msg}; falling back to psutil", file=sys.stderr)
                    monitor_proc = None
                    monitor_mode = "psutil"
                    monitor_thread = _start_psutil_sampler(
                        interval_ms=args.interval_ms,
                        writer=writer,
                        csv_file=csv_file,
                        stop_event=stop_event,
                        sample_counter=sample_counter,
                    )
                    monitor_stderr = err
                else:
                    raise RuntimeError(msg)
        elif args.fallback_psutil:
            monitor_mode = "psutil"
            monitor_thread = _start_psutil_sampler(
                interval_ms=args.interval_ms,
                writer=writer,
                csv_file=csv_file,
                stop_event=stop_event,
                sample_counter=sample_counter,
            )
        else:
            raise RuntimeError(
                "Jetson tegrastats monitoring unavailable. "
                "Use --fallback-psutil for desktop RAM/swap/CPU-only monitoring."
            )

        run_ingest_path = Path(__file__).resolve().parent / "run_ingest.py"
        cmd = [sys.executable, str(run_ingest_path), *args.ingest_args]
        print(f"[monitor] launching: {' '.join(cmd)}", file=sys.stderr)
        ingest_proc = subprocess.Popen(cmd)
        ingest_exit = ingest_proc.wait()

        stop_event.set()
        if monitor_proc is not None:
            if monitor_proc.poll() is None:
                monitor_proc.terminate()
                try:
                    monitor_proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    monitor_proc.kill()
                    monitor_proc.wait(timeout=3.0)
            if monitor_proc.stderr is not None:
                try:
                    monitor_stderr = (monitor_stderr + "\n" + monitor_proc.stderr.read()).strip()
                except Exception:
                    pass
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)

    wrote_plot = _write_summary_plot(metrics_csv, summary_png)
    wrote_html = False
    if args.html and wrote_plot:
        _write_summary_html(summary_png, summary_html)
        wrote_html = True

    zip_path = ""
    if args.zip:
        zip_path = shutil.make_archive(str(output_dir), "zip", output_dir)

    end_time = _now_iso()
    metadata: dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
        "sample_interval_ms": args.interval_ms,
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_csv),
        "summary_png": str(summary_png if wrote_plot else ""),
        "summary_html": str(summary_html if wrote_html else ""),
        "zip_path": zip_path,
        "ingest_command": [sys.executable, "run_ingest.py", *args.ingest_args],
        "ingest_exit_code": ingest_exit,
        "monitor_mode": monitor_mode,
        "is_jetson": is_jetson,
        "tegrastats_path_requested": args.tegrastats_path,
        "tegrastats_path_resolved": tegrastats_resolved or "",
        "tegrastats_stderr": monitor_stderr,
        "samples_written": sample_counter[0],
        "fallback_psutil": bool(args.fallback_psutil),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[monitor] wrote {metrics_csv}", file=sys.stderr)
    print(f"[monitor] wrote {metadata_json}", file=sys.stderr)
    if wrote_plot:
        print(f"[monitor] wrote {summary_png}", file=sys.stderr)
    if wrote_html:
        print(f"[monitor] wrote {summary_html}", file=sys.stderr)
    if zip_path:
        print(f"[monitor] wrote {zip_path}", file=sys.stderr)
    return int(ingest_exit)


if __name__ == "__main__":
    raise SystemExit(main())
