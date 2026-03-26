#!/usr/bin/env python3
"""
Install missing pipeline dependencies based on jetson_env_report.json (Step 2).

If overall_ok is true: no-op. Otherwise pip-installs only packages that failed import
checks. May add numpy>=1.26,<2 and scipy/scikit-learn when NumPy 2.x / SciPy ABI issues
are detected. Does not put `torch` on the pip install list; when the report says torch imports OK,
passes `-c` constraints so pip does not replace Jetson torch with a PyPI CUDA wheel.
Otherwise prints NVIDIA Jetson PyTorch guidance.

Re-run check_jetson_pipeline_env.py after this script (until overall_ok).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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


def _torch_pip_constraint_args(report: dict) -> tuple[list[str], Path | None]:
    """If torch already imports OK, pin it so pip never swaps Jetson torch for PyPI CUDA wheels."""
    t = (report.get("packages") or {}).get("torch") or {}
    if t.get("ok") is not True:
        return [], None
    ver = (t.get("version") or "").strip()
    if not ver:
        return [], None
    fd, raw = tempfile.mkstemp(suffix="-torch-constraint.txt", text=True)
    path = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"torch=={ver}\n")
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        return [], None
    return ["-c", str(path)], path


def _numpy_major(version: str | None) -> int | None:
    if not version:
        return None
    try:
        return int(str(version).split(".")[0])
    except (ValueError, IndexError):
        return None


def _needs_numpy_pin(report: dict) -> bool:
    """Jetson: NumPy 2.x often breaks apt SciPy; pin to 1.x for sklearn/sentence_transformers."""
    p = (report.get("packages") or {}).get("numpy") or {}
    if not p.get("ok"):
        return True
    maj = _numpy_major(p.get("version"))
    return maj is not None and maj >= 2


def _needs_scipy_stack(report: dict) -> bool:
    st = (report.get("packages") or {}).get("sentence_transformers") or {}
    err = (st.get("error") or "") + str(st.get("message") or "")
    if any(
        x in err
        for x in (
            "_ARRAY_API",
            "multiarray",
            "scipy",
            "sklearn",
            "numpy.core",
            "dtype size changed",
            "binary incompatibility",
        )
    ):
        return True
    return bool(st.get("ok") is False and _needs_numpy_pin(report))


def _needs_hf_stack_compat_upgrade(report: dict) -> bool:
    """transformers 5.x + older sentence-transformers often breaks dynamic PreTrainedModel imports."""
    st = (report.get("packages") or {}).get("sentence_transformers") or {}
    err = (st.get("error") or "") + str(st.get("message") or "")
    return any(x in err for x in ("PreTrainedModel", "requirements defined correctly"))


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

    if _needs_numpy_pin(report):
        specs = [s for s in specs if s not in ("numpy", "numpy>=1.26,<2")]
        specs.insert(0, "numpy>=1.26,<2")
    if _needs_scipy_stack(report):
        for extra in ("scipy", "scikit-learn"):
            if extra not in specs:
                specs.append(extra)
    return specs


NUMPY_SCIPY_PRUNE = frozenset({"numpy", "numpy>=1.26,<2", "scipy", "scikit-learn"})


def _split_numpy_phase(report: dict, specs: list[str]) -> tuple[list[str], list[str]]:
    """First phase: pin NumPy 1.x + pip SciPy/sklearn so ABI matches before HF packages."""
    if not (_needs_numpy_pin(report) or _needs_scipy_stack(report)):
        return [], specs
    phase = ["numpy>=1.26,<2", "scipy", "scikit-learn"]
    rest = [s for s in specs if s not in NUMPY_SCIPY_PRUNE]
    return phase, rest


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

    constraint_args, constraint_path = _torch_pip_constraint_args(report)

    def _cleanup_constraint() -> None:
        if constraint_path is not None:
            try:
                constraint_path.unlink(missing_ok=True)
            except OSError:
                pass

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

    ran_numpy_phase = False
    phase_numpy, specs = _split_numpy_phase(report, specs)
    if phase_numpy:
        ran_numpy_phase = True
        np_entry = (report.get("packages") or {}).get("numpy") or {}
        force = _needs_numpy_pin(report) and (_numpy_major(np_entry.get("version")) or 0) >= 2
        np_cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if force:
            np_cmd.extend(["--force-reinstall", "--no-cache-dir"])
        np_cmd.extend(constraint_args)
        np_cmd.extend(phase_numpy)
        print("Running (NumPy/SciPy stack first — fixes dtype / sklearn ABI on Jetson):", " ".join(np_cmd))
        np_proc = subprocess.run(np_cmd, check=False)
        if np_proc.returncode != 0:
            print("NumPy/SciPy install failed.", file=sys.stderr)
            _cleanup_constraint()
            return np_proc.returncode

    ran_hf_upgrade = False
    if _needs_hf_stack_compat_upgrade(report):
        ran_hf_upgrade = True
        up_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            *constraint_args,
            "sentence-transformers",
            "transformers",
            "huggingface-hub",
        ]
        print(
            "Running (HF stack force-reinstall for PreTrainedModel / ST compatibility):",
            " ".join(up_cmd),
        )
        up_proc = subprocess.run(up_cmd, check=False)
        if up_proc.returncode != 0:
            print("HF stack upgrade failed.", file=sys.stderr)
            _cleanup_constraint()
            return up_proc.returncode
        prune = {"sentence-transformers", "transformers", "huggingface-hub"}
        specs = [s for s in specs if s not in prune]

    if not specs:
        if ran_hf_upgrade or ran_numpy_phase:
            print(
                "Pip phases finished. Re-run: python3 check_jetson_pipeline_env.py",
                file=sys.stderr,
            )
            _cleanup_constraint()
            return 0
        print(
            "No automatable pip packages listed as failed imports; "
            "fix manual steps above, then re-run the check script.",
            file=sys.stderr,
        )
        _cleanup_constraint()
        return 1

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    # After pinning NumPy 1.x, wheels built against NumPy 2 (e.g. sentence-transformers,
    # tokenizers) must be reinstalled or pip leaves "Requirement already satisfied".
    if ran_numpy_phase:
        cmd.extend(["--force-reinstall", "--no-cache-dir"])
    cmd.extend(constraint_args)
    cmd.extend(specs)
    print("Running:", " ".join(cmd))

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print("pip install failed.", file=sys.stderr)
        _cleanup_constraint()
        return proc.returncode

    print("pip install finished. Re-run: python3 check_jetson_pipeline_env.py")
    _cleanup_constraint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
