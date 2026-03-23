#!/usr/bin/env python3
"""
Install missing pipeline dependencies based on jetson_env_report.json (Step 2).

If overall_ok is true: no-op. Otherwise pip-installs only packages that failed import
checks. Does not guess a generic torch wheel; prints NVIDIA Jetson PyTorch guidance.

Re-run check_jetson_pipeline_env.py after this script (until overall_ok).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_REPORT = Path(__file__).resolve().parent / "jetson_env_report.json"

# pip packages to install when the corresponding import failed (from report.packages)
PACKAGE_KEYS = (
    "torch",
    "transformers",
    "sentence_transformers",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "tokenizers",
)

JETSON_PYTORCH_HINT = """\
Jetson PyTorch must match JetPack (do not use generic pip torch on device).
See: https://developer.nvidia.com/embedded/downloads
Search for PyTorch for Jetson / JetPack 6.x and install the provided wheel or index URL,
then re-run check_jetson_pipeline_env.py.
"""


def load_report(path: Path) -> dict:
    if not path.is_file():
        print(f"error: report not found: {path}", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def pip_specs_to_install(report: dict) -> list[str]:
    """Collect pip install targets for failed imports. Never auto-install `torch` (use NVIDIA Jetson wheel)."""
    specs: list[str] = []
    packages = report.get("packages") or {}
    for key in PACKAGE_KEYS:
        if key == "torch":
            continue
        entry = packages.get(key)
        if not entry:
            continue
        if entry.get("ok") is True:
            continue
        pip_spec = entry.get("pip_spec") or key
        if pip_spec not in specs:
            specs.append(pip_spec)
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_path",
        type=Path,
        nargs="?",
        default=DEFAULT_REPORT,
        help=f"path to jetson_env_report.json (default: {DEFAULT_REPORT})",
    )
    args = parser.parse_args()
    report = load_report(args.report_path)

    if report.get("overall_ok") is True:
        print("All pipeline prerequisites satisfied per report; nothing to install.")
        return 0

    specs = pip_specs_to_install(report)
    cuda = report.get("cuda") or {}
    torch_entry = (report.get("packages") or {}).get("torch") or {}

    if torch_entry.get("ok") is not True:
        print("torch import failed - install Jetson-matched PyTorch from NVIDIA (not generic pip).", file=sys.stderr)
        print(JETSON_PYTORCH_HINT, file=sys.stderr)
    elif cuda.get("torch_cuda_available") is not True:
        print(JETSON_PYTORCH_HINT, file=sys.stderr)

    model = report.get("model_voyage_4_nano") or {}
    if model.get("status") == "not_in_cache":
        print(
            "Model not in Hugging Face cache. After torch/CUDA work, prefetch with e.g.:\n"
            "  huggingface-cli download voyageai/voyage-4-nano\n"
            "or: python -c \"from huggingface_hub import snapshot_download; "
            'snapshot_download(\\"voyageai/voyage-4-nano\\")\"',
            file=sys.stderr,
        )

    if not specs:
        print(
            "No automatable pip packages listed as failed imports; "
            "fix manual steps above, then re-run the check script.",
            file=sys.stderr,
        )
        return 1

    cmd = [sys.executable, "-m", "pip", "install", *specs]
    print("Running:", " ".join(cmd))

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print("pip install failed.", file=sys.stderr)
        return proc.returncode

    print("pip install finished. Re-run: python3 check_jetson_pipeline_env.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
