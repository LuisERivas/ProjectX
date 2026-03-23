#!/usr/bin/env python3
"""
Read-only prerequisite check for the embedding pipeline (Step 2).

Does not run pip, download models, or mutate system state. Probes JetPack, Python,
CUDA/torch, Python packages, and voyage-4-nano loadability (local cache only).

See embeddingCreationPlan.txt Step 2. Output: JSON file + stdout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MODEL_ID = "voyageai/voyage-4-nano"
DEFAULT_REPORT = Path(__file__).resolve().parent / "jetson_env_report.json"
TEGRA_RELEASE = Path("/etc/nv_tegra_release")

# import_name -> pip install argument
PIP_BY_IMPORT = {
    "torch": "torch",
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "tokenizers": "tokenizers",
}


def _probe_jetpack() -> dict[str, Any]:
    if platform.system() != "Linux":
        return {
            "status": "skipped",
            "ok": True,
            "message": "Not Linux; JetPack file check skipped (run on Jetson for real probe).",
        }
    if not TEGRA_RELEASE.is_file():
        arch = platform.machine().lower()
        if arch in ("aarch64", "arm64"):
            return {
                "status": "missing",
                "ok": False,
                "message": f"{TEGRA_RELEASE} not found; expected on Jetson.",
            }
        return {
            "status": "skipped",
            "ok": True,
            "message": "Linux non-aarch64: not treating as Jetson.",
        }
    try:
        text = TEGRA_RELEASE.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as e:
        return {"status": "error", "ok": False, "message": str(e)}
    return {
        "status": "ok",
        "ok": True,
        "path": str(TEGRA_RELEASE),
        "content_preview": text[:500] + ("..." if len(text) > 500 else ""),
    }


def _probe_python() -> dict[str, Any]:
    return {
        "ok": True,
        "version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def _probe_import(name: str) -> dict[str, Any]:
    pip = PIP_BY_IMPORT.get(name, name)
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", None)
        return {"ok": True, "version": ver, "pip_spec": pip}
    except Exception as e:  # noqa: BLE001 — report any import failure
        return {"ok": False, "error": repr(e), "pip_spec": pip}


def _probe_cuda() -> dict[str, Any]:
    t = _probe_import("torch")
    if not t.get("ok"):
        return {
            "ok": False,
            "message": "torch not importable; CUDA probe skipped.",
            "torch_import": t,
        }
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": repr(e)}

    avail = torch.cuda.is_available()
    out: dict[str, Any] = {
        "ok": avail,
        "torch_cuda_available": avail,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if avail else 0,
    }
    if avail and torch.cuda.device_count() > 0:
        try:
            out["device_0_name"] = torch.cuda.get_device_name(0)
        except Exception as e:  # noqa: BLE001
            out["device_0_name_error"] = repr(e)
    if not avail:
        out["message"] = "torch.cuda.is_available() is False"
    return out


def _model_in_hf_cache() -> bool:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return False
    try:
        info = scan_cache_dir()
    except Exception:
        return False
    for repo in getattr(info, "repos", []) or []:
        rid = getattr(repo, "repo_id", None)
        if rid == MODEL_ID:
            return True
    return False


def _probe_model_voyage() -> dict[str, Any]:
    st = _probe_import("sentence_transformers")
    if not st.get("ok"):
        return {
            "ok": False,
            "message": "sentence_transformers not importable.",
            "sentence_transformers_import": st,
            "pip_automatable": True,
        }
    tr = _probe_import("torch")
    if not tr.get("ok"):
        return {
            "ok": False,
            "message": "torch not importable.",
            "pip_automatable": True,
        }

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": repr(e), "pip_automatable": True}

    if not torch.cuda.is_available():
        return {
            "ok": False,
            "status": "cuda_required",
            "message": "CUDA not available; Step 2 model probe requires GPU.",
            "pip_automatable": False,
        }

    if not _model_in_hf_cache():
        return {
            "ok": False,
            "status": "not_in_cache",
            "message": (
                f"{MODEL_ID} not found in local Hugging Face cache. "
                "Prefetch with: huggingface-cli download voyageai/voyage-4-nano "
                "or snapshot_download (network once)."
            ),
            "pip_automatable": False,
        }

    try:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "truncate_dim": 2048,
            "model_kwargs": {
                "local_files_only": True,
                "attn_implementation": "sdpa",
                "torch_dtype": torch.bfloat16,
            },
        }
        model = SentenceTransformer(MODEL_ID, **kwargs)
        dev = model.device
        return {
            "ok": True,
            "status": "loaded_local_only",
            "device": str(dev),
            "message": "SentenceTransformer constructed with local_files_only=True.",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "status": "load_failed",
            "message": repr(e),
            "pip_automatable": False,
        }


def _probe_soft_icu() -> dict[str, Any]:
    for mod in ("icu", "PyICU"):
        try:
            importlib.import_module(mod)
            return {
                "ok": True,
                "module": mod,
                "not_required_for_step2": True,
                "message": "ICU bindings present (optional for Step 4).",
            }
        except Exception:
            continue
    return {
        "ok": True,
        "not_required_for_step2": True,
        "message": "No icu/PyICU (optional until Step 4).",
    }


def _disk_note() -> dict[str, Any]:
    import os

    env_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        return {"hf_hub_cache": env_cache, "note": "From environment."}
    try:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
        return {
            "hf_hub_cache": str(cache),
            "note": "Inspect free space outside this script if needed.",
        }
    except Exception:
        return {"note": "huggingface_hub not installed or HF cache path unknown."}


def build_report() -> dict[str, Any]:
    jetpack = _probe_jetpack()
    python_info = _probe_python()
    packages = {name: _probe_import(name) for name in PIP_BY_IMPORT}
    cuda = _probe_cuda()
    model = _probe_model_voyage()
    soft = {"icu": _probe_soft_icu()}
    disk = _disk_note()

    packages_ok = all(p.get("ok") for p in packages.values())
    cuda_ok = cuda.get("ok") is True
    model_ok = model.get("ok") is True

    overall_ok = (
        jetpack.get("ok") is True
        and python_info.get("ok") is True
        and packages_ok
        and cuda_ok
        and model_ok
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jetpack": jetpack,
        "python": python_info,
        "packages": packages,
        "cuda": cuda,
        "model_voyage_4_nano": model,
        "soft_future": soft,
        "disk": disk,
        "overall_ok": overall_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JSON report path (default: {DEFAULT_REPORT})",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote report to: {args.output.resolve()}", file=sys.stderr)
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
