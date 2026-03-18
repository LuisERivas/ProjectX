#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_card3_validation.sh [--repo-root PATH] [--build-dir PATH] [--out-dir PATH] [--runs N] [--warmup-runs N]

Description:
  Runs Card 3 validation end-to-end:
    1) targeted Card 3 correctness/compliance/event tests
    2) Card 3 residency A/B benchmark
    3) PASS/FAIL evaluation against Card 3 expectations

Notes:
  - Requires an already-built CLI/test binary set in --build-dir.
  - This script does NOT modify contracts or plan files.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 2
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${DEFAULT_REPO_ROOT}"
BUILD_DIR=""
OUT_DIR=""
RUNS="10"
WARMUP_RUNS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --warmup-runs)
      WARMUP_RUNS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

VDB_DIR="${REPO_ROOT}/vector_db_v3"
if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${VDB_DIR}/build"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${VDB_DIR}/gate_evidence/card3_validation"
fi

require_cmd python3
require_cmd ctest

mkdir -p "${OUT_DIR}"

CLI="${BUILD_DIR}/vectordb_v3_cli"
if [[ ! -x "${CLI}" ]]; then
  CLI="${BUILD_DIR}/vectordb_v3_cli.exe"
fi
if [[ ! -x "${CLI}" ]]; then
  echo "error: vectordb_v3_cli not found in ${BUILD_DIR}" >&2
  exit 2
fi

echo "== Card 3 Validation =="
echo "Repo root : ${REPO_ROOT}"
echo "Build dir : ${BUILD_DIR}"
echo "Out dir   : ${OUT_DIR}"
echo "Runs      : ${RUNS} (warmup=${WARMUP_RUNS})"
echo

TEST_REGEX="vectordb_v3_kmeans_backend_parity_tests|vectordb_v3_kmeans_tie_break_determinism_tests|vectordb_v3_kmeans_backend_selection_tests|vectordb_v3_gpu_residency_tests|vectordb_v3_compliance_pass_tests|vectordb_v3_compliance_fail_fast_tests|vectordb_v3_terminal_event_contract_tests"

echo "[1/3] Running targeted Card 3 tests"
ctest \
  --test-dir "${BUILD_DIR}" \
  --output-on-failure \
  -R "${TEST_REGEX}" | tee "${OUT_DIR}/card3_targeted_ctest.log"

echo "[2/3] Running Card 3 A/B benchmark"
python3 "${VDB_DIR}/scripts/card3_residency_ab.py" \
  --build-dir "${BUILD_DIR}" \
  --out-dir "${OUT_DIR}" \
  --runs "${RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" | tee "${OUT_DIR}/card3_ab_stdout.log"

echo "[3/3] Evaluating PASS/FAIL checklist"
AB_JSON="${OUT_DIR}/card3_residency_ab.json"
python3 - <<PY
import json
import sys
from pathlib import Path

path = Path("${AB_JSON}")
if not path.exists():
    print(f"FAIL: missing A/B output file: {path}", file=sys.stderr)
    sys.exit(1)

obj = json.loads(path.read_text(encoding="utf-8"))
modes = obj.get("modes", {})
base = modes.get("baseline_off", {}).get("summary", {})
cand = modes.get("candidate_stage", {}).get("summary", {})
delta = obj.get("delta", {})

top_imp = float(delta.get("top_median_improvement_pct", 0.0))
mid_imp = float(delta.get("mid_median_improvement_pct", 0.0))
base_alloc = float(base.get("alloc_calls_median", 0.0))
cand_alloc = float(cand.get("alloc_calls_median", 0.0))
cand_h2d_saved = float(cand.get("h2d_saved_est_median", 0.0))
cand_hits = float(cand.get("cache_hits_median", 0.0))

checks = []
checks.append(("allocation_non_regression", cand_alloc <= base_alloc))
checks.append(("residency_signal_present", cand_h2d_saved > 0 or cand_hits > 0))
checks.append(("latency_improvement_top_or_mid", top_imp > 0.0 or mid_imp > 0.0))

print("Card 3 checklist:")
print(f"- top_median_improvement_pct: {top_imp:.3f}")
print(f"- mid_median_improvement_pct: {mid_imp:.3f}")
print(f"- baseline_alloc_calls_median: {base_alloc:.3f}")
print(f"- candidate_alloc_calls_median: {cand_alloc:.3f}")
print(f"- candidate_h2d_saved_est_median: {cand_h2d_saved:.3f}")
print(f"- candidate_cache_hits_median: {cand_hits:.3f}")

failed = [name for name, ok in checks if not ok]
if failed:
    print(f"FAIL: checklist failed: {', '.join(failed)}", file=sys.stderr)
    print(f"Evidence: ${AB_JSON}", file=sys.stderr)
    sys.exit(1)

print("PASS: Card 3 behavior validated.")
PY

echo
echo "Card 3 validation complete."
echo "Evidence:"
echo "  - ${OUT_DIR}/card3_targeted_ctest.log"
echo "  - ${OUT_DIR}/card3_ab_stdout.log"
echo "  - ${OUT_DIR}/card3_residency_ab.json"
