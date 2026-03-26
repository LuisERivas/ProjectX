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
import subprocess
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


def _distribution_version(dist_name: str) -> str | None:
    """PyPI distribution version even when `import module` fails (broken/cached install)."""
    code = (
        "import sys\n"
        "try:\n"
        "    from importlib.metadata import version\n"
        f"    print(version({dist_name!r}))\n"
        "except Exception as e:\n"
        "    print(repr(e), file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out if out else None


def _probe_import(name: str) -> dict[str, Any]:
    """Import in a **subprocess** so a broken SciPy/NumPy/sklearn chain cannot crash this script."""
    pip = PIP_BY_IMPORT.get(name, name)
    code = (
        "import importlib, sys\n"
        f"try:\n"
        f"    m = importlib.import_module({name!r})\n"
        f"    v = getattr(m, '__version__', '') or ''\n"
        f"    print(v)\n"
        f"except Exception as e:\n"
        f"    print(repr(e), file=sys.stderr)\n"
        f"    sys.exit(1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        out: dict[str, Any] = {"ok": False, "error": err[-8000:], "pip_spec": pip}
        dist_ver = _distribution_version(pip)
        if dist_ver:
            out["distribution_version"] = dist_ver
        return out
    ver = (proc.stdout or "").strip()
    return {"ok": True, "version": ver if ver else None, "pip_spec": pip}


def _probe_pillow() -> dict[str, Any]:
    """Debian apt Pillow is often <9.1 and lacks PIL.Image.Resampling; transformers 5.x then breaks PreTrainedModel import."""
    code = (
        "import json, sys\n"
        "try:\n"
        "    from PIL import Image\n"
        "    from importlib.metadata import version, PackageNotFoundError\n"
        "    try:\n"
        "        pv = version('pillow')\n"
        "    except PackageNotFoundError:\n"
        "        pv = None\n"
        "    has_resampling = hasattr(Image, 'Resampling')\n"
        "    out = {\n"
        "        'ok': has_resampling,\n"
        "        'version': pv,\n"
        "        'pip_spec': 'pillow',\n"
        "        'pil_image_file': getattr(Image, '__file__', None),\n"
        "    }\n"
        "    if not has_resampling:\n"
        "        out['error'] = (\n"
        "            'PIL.Image.Resampling missing (system Pillow too old for transformers 5.x). '\n"
        "            'pip install --user --upgrade \"pillow>=9.1\"'\n"
        "        )\n"
        "    print(json.dumps(out))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'error': repr(e), 'pip_spec': 'pillow'}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    raw = (proc.stdout or "").strip()
    line = raw.splitlines()[-1] if raw else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": (raw or (proc.stderr or ""))[-4000:],
            "pip_spec": "pillow",
        }


def _probe_cuda() -> dict[str, Any]:
    """Run CUDA probe in a subprocess (isolated from package import order in the parent)."""
    code = (
        "import json, sys\n"
        "try:\n"
        "    import torch\n"
        "except Exception as e:\n"
        "    print(json.dumps({\n"
        "        'ok': False,\n"
        "        'message': 'torch not importable; CUDA probe skipped.',\n"
        "        'torch_import': {'ok': False, 'error': repr(e)},\n"
        "    }))\n"
        "    sys.exit(0)\n"
        "avail = torch.cuda.is_available()\n"
        "out = {\n"
        "    'ok': avail,\n"
        "    'torch_cuda_available': avail,\n"
        "    'torch_version': torch.__version__,\n"
        "    'torch_cuda_version': getattr(torch.version, 'cuda', None),\n"
        "    'device_count': torch.cuda.device_count() if avail else 0,\n"
        "}\n"
        "if avail and torch.cuda.device_count() > 0:\n"
        "    try:\n"
        "        out['device_0_name'] = torch.cuda.get_device_name(0)\n"
        "    except Exception as e:\n"
        "        out['device_0_name_error'] = repr(e)\n"
        "if not avail:\n"
        "    out['message'] = 'torch.cuda.is_available() is False'\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        return {
            "ok": False,
            "message": "CUDA subprocess produced no output",
            "stderr": (proc.stderr or "")[-2000:],
        }
    try:
        return json.loads(raw.splitlines()[-1])
    except json.JSONDecodeError:
        return {
            "ok": False,
            "message": "failed to parse CUDA probe JSON",
            "stdout": raw[-4000:],
            "stderr": (proc.stderr or "")[-2000:],
        }


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


def _probe_model_voyage_subprocess_load() -> dict[str, Any]:
    """Load SentenceTransformer in a subprocess so import failures never kill the checker."""
    code = (
        "import json, sys, torch\n"
        "from sentence_transformers import SentenceTransformer\n"
        f"MODEL_ID = {MODEL_ID!r}\n"
        "kwargs = {\n"
        "    'trust_remote_code': True,\n"
        "    'truncate_dim': 2048,\n"
        "    'model_kwargs': {\n"
        "        'local_files_only': True,\n"
        "        'attn_implementation': 'sdpa',\n"
        "        'torch_dtype': torch.bfloat16,\n"
        "    },\n"
        "}\n"
        "try:\n"
        "    model = SentenceTransformer(MODEL_ID, **kwargs)\n"
        "    print(json.dumps({\n"
        "        'ok': True,\n"
        "        'status': 'loaded_local_only',\n"
        "        'device': str(model.device),\n"
        "        'message': 'SentenceTransformer with local_files_only=True.',\n"
        "    }))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'status': 'load_failed', 'message': repr(e)}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    raw = (proc.stdout or "").strip()
    line = raw.splitlines()[-1] if raw else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "status": "load_failed",
            "message": "subprocess load parse error",
            "stdout": raw[-4000:],
            "stderr": (proc.stderr or "")[-2000:],
        }


def _probe_model_voyage(cuda: dict[str, Any]) -> dict[str, Any]:
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

    if cuda.get("torch_cuda_available") is not True:
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

    return _probe_model_voyage_subprocess_load()


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


def _environment_warnings(
    jetpack: dict[str, Any],
    packages: dict[str, Any],
    cuda: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    np_entry = packages.get("numpy") or {}
    if np_entry.get("ok") and np_entry.get("version"):
        try:
            major = int(str(np_entry["version"]).split(".")[0])
            if major >= 2:
                warnings.append(
                    "NumPy 2.x with Debian/apt SciPy in /usr/lib/python3/dist-packages often breaks "
                    "sentence_transformers/sklearn (AttributeError _ARRAY_API, numpy.core.multiarray, "
                    "or ValueError numpy.dtype size changed / binary incompatibility). "
                    "Fix: pip install --user 'numpy>=1.26,<2' 'scipy' 'scikit-learn' --upgrade, "
                    "then re-run this check."
                )
        except (ValueError, IndexError):
            pass
    tv = (cuda or {}).get("torch_version") or ""
    if tv and "+cpu" in str(tv) and jetpack.get("status") == "ok":
        warnings.append(
            "PyTorch is a PyPI CPU-only build (+cpu). On Jetson, uninstall it and install the NVIDIA "
            "Jetson CUDA PyTorch wheel for your JetPack from https://developer.nvidia.com/embedded/downloads — "
            "otherwise torch.cuda.is_available() stays false."
        )
    if tv and "+cu" in str(tv) and jetpack.get("status") == "ok":
        if not (cuda or {}).get("torch_cuda_available"):
            warnings.append(
                "PyTorch looks like a desktop CUDA wheel (e.g. +cu130). On Jetson, use the NVIDIA "
                "Jetson PyTorch build for your JetPack from https://developer.nvidia.com/embedded/downloads — "
                "PyPI CUDA wheels often mismatch the Jetson driver and report CUDA init errors."
            )
    st_err = ((packages.get("sentence_transformers") or {}).get("error") or "") + str(
        (packages.get("sentence_transformers") or {}).get("message") or ""
    )
    pl = (packages.get("pillow") or {})
    if pl.get("ok") is not True:
        warnings.append(
            "Pillow (PIL) is missing or too old for transformers 5.x — Debian apt Pillow often lacks "
            "PIL.Image.Resampling, which surfaces as ModuleNotFoundError PreTrainedModel. "
            "Fix: python3 -m pip install --user --upgrade 'pillow>=9.1' (then re-run this check)."
        )
    elif any(x in st_err for x in ("PreTrainedModel", "requirements defined correctly")):
        warnings.append(
            "sentence_transformers failed to import PreTrainedModel / dynamic HF modules — often broken "
            "or mixed installs (apt python3-* plus pip, or stale wheels). Fix: remove conflicting apt "
            "packages if any (`apt list --installed | grep -i sentence`); then "
            "`python3 -m pip install --upgrade --force-reinstall --no-cache-dir sentence-transformers "
            "transformers huggingface-hub`. Or run install_jetson_pipeline_deps.py (uses force-reinstall "
            "for this error)."
        )
    return warnings


def build_report() -> dict[str, Any]:
    jetpack = _probe_jetpack()
    python_info = _probe_python()
    packages = {name: _probe_import(name) for name in PIP_BY_IMPORT}
    packages["pillow"] = _probe_pillow()
    cuda = _probe_cuda()
    model = _probe_model_voyage(cuda)
    soft = {"icu": _probe_soft_icu()}
    disk = _disk_note()
    env_warnings = _environment_warnings(jetpack, packages, cuda)

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
        "environment_warnings": env_warnings,
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
