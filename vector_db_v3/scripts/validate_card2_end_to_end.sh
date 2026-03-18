#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  validate_card2_end_to_end.sh [--repo-root PATH] [--build-dir PATH] [--data-dir PATH] [--keep-build]

Description:
  Runs the full Card 2 validation flow:
    1) clean Release configure + build
    2) Card-2 focused ctest suites
    3) backend selection spot checks (tensor/fp32/auto)
    4) terminal + CLI contract python tests
    5) minimal gate run (G1/G3/G5/G6/G7 profile)
    6) reproducibility events-only check
    7) perf evidence capture

Outputs:
  Writes timestamped evidence under:
    vector_db_v3/gate_evidence/card2_validation_<UTC timestamp>/
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
DATA_DIR=""
KEEP_BUILD="0"

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
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --keep-build)
            KEEP_BUILD="1"
            shift
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
if [[ -z "${DATA_DIR}" ]]; then
    DATA_DIR="/tmp/vdb_card2_validation"
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="${VDB_DIR}/gate_evidence/card2_validation_${TS}"
mkdir -p "${EVIDENCE_DIR}"

echo "== Card 2 Validation =="
echo "Repo root : ${REPO_ROOT}"
echo "VDB dir   : ${VDB_DIR}"
echo "Build dir : ${BUILD_DIR}"
echo "Data dir  : ${DATA_DIR}"
echo "Evidence  : ${EVIDENCE_DIR}"
echo

require_cmd cmake
require_cmd ctest
require_cmd python3

if [[ ! -d "${VDB_DIR}" ]]; then
    echo "error: vector_db_v3 directory not found under repo root: ${VDB_DIR}" >&2
    exit 2
fi

if [[ "${KEEP_BUILD}" != "1" ]]; then
    echo "[1/7] Clean configure + build (Release)"
    rm -rf "${BUILD_DIR}"
else
    echo "[1/7] Reusing existing build directory"
fi

cmake -S "${VDB_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j
echo

CLI="${BUILD_DIR}/vectordb_v3_cli"
if [[ ! -x "${CLI}" ]]; then
    CLI="${BUILD_DIR}/vectordb_v3_cli.exe"
fi
if [[ ! -x "${CLI}" ]]; then
    echo "error: could not find built CLI binary in ${BUILD_DIR}" >&2
    exit 2
fi

echo "[2/7] Card-2 focused test suites"
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R \
"vectordb_v3_kmeans_backend_parity_tests|vectordb_v3_kmeans_tie_break_determinism_tests|vectordb_v3_kmeans_backend_selection_tests|vectordb_v3_compliance_pass_tests|vectordb_v3_compliance_fail_fast_tests"
echo

echo "[3/7] Backend selection spot checks"
rm -rf "${DATA_DIR}"
"${CLI}" init --path "${DATA_DIR}" > "${EVIDENCE_DIR}/init.json"
"${CLI}" insert --path "${DATA_DIR}" --id 1 --vec "$(python3 - <<'PY'
print(",".join(["0.1"] * 1024))
PY
)" > "${EVIDENCE_DIR}/insert.json"

env VECTOR_DB_V3_KMEANS_BACKEND=cuda VECTOR_DB_V3_KMEANS_PRECISION=tensor \
    "${CLI}" build-top-clusters --path "${DATA_DIR}" --seed 7 > "${EVIDENCE_DIR}/build_top_tensor.jsonl"
env VECTOR_DB_V3_KMEANS_BACKEND=cuda VECTOR_DB_V3_KMEANS_PRECISION=tensor \
    "${CLI}" cluster-stats --path "${DATA_DIR}" > "${EVIDENCE_DIR}/cluster_stats_tensor.json"

env VECTOR_DB_V3_KMEANS_BACKEND=cuda VECTOR_DB_V3_KMEANS_PRECISION=fp32 \
    "${CLI}" build-top-clusters --path "${DATA_DIR}" --seed 7 > "${EVIDENCE_DIR}/build_top_fp32.jsonl"
env VECTOR_DB_V3_KMEANS_BACKEND=cuda VECTOR_DB_V3_KMEANS_PRECISION=fp32 \
    "${CLI}" cluster-stats --path "${DATA_DIR}" > "${EVIDENCE_DIR}/cluster_stats_fp32.json"

env VECTOR_DB_V3_KMEANS_BACKEND=auto VECTOR_DB_V3_KMEANS_PRECISION=auto \
    "${CLI}" build-top-clusters --path "${DATA_DIR}" --seed 7 > "${EVIDENCE_DIR}/build_top_auto.jsonl"
env VECTOR_DB_V3_KMEANS_BACKEND=auto VECTOR_DB_V3_KMEANS_PRECISION=auto \
    "${CLI}" cluster-stats --path "${DATA_DIR}" > "${EVIDENCE_DIR}/cluster_stats_auto.json"

python3 - <<PY
import json, pathlib
root = pathlib.Path("${EVIDENCE_DIR}")
for name in ["cluster_stats_tensor.json", "cluster_stats_fp32.json", "cluster_stats_auto.json"]:
    p = root / name
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"{name}: backend={data.get('kernel_backend_path')} tensor_active={data.get('tensor_core_active')} compliance={data.get('compliance_status')}")
PY
echo

echo "[4/7] Contract checks (terminal + CLI)"
python3 "${VDB_DIR}/scripts/test_terminal_event_contract.py"
python3 "${VDB_DIR}/scripts/test_cli_contract.py"
echo

echo "[5/7] Minimal gate run"
python3 "${VDB_DIR}/scripts/run_gates.py" \
    --build-dir "${BUILD_DIR}" \
    --profile minimal \
    --run-id "card2_validation_minimal_${TS}"
echo

echo "[6/7] Reproducibility check"
python3 "${VDB_DIR}/scripts/check_reproducibility.py" \
    --build-dir "${BUILD_DIR}" \
    --events-only \
    --out-dir "${EVIDENCE_DIR}/repro"
echo

echo "[7/7] Performance evidence capture"
python3 "${VDB_DIR}/scripts/perf_gate.py" \
    --build-dir "${BUILD_DIR}" \
    --mode evidence \
    --profile minimum \
    --warmup-runs 1 \
    --measure-runs 5 \
    --out-dir "${EVIDENCE_DIR}/perf"
echo

echo "Card 2 end-to-end validation completed."
echo "Evidence directory: ${EVIDENCE_DIR}"
echo "Key files:"
echo "  - ${EVIDENCE_DIR}/cluster_stats_tensor.json"
echo "  - ${EVIDENCE_DIR}/cluster_stats_fp32.json"
echo "  - ${EVIDENCE_DIR}/perf/perf_metrics.json"
echo "  - ${VDB_DIR}/gate_evidence/card2_validation_minimal_${TS}/summary.json"
