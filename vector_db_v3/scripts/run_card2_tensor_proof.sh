#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_card2_tensor_proof.sh [--repo-root PATH] [--build-dir PATH] [--data-dir PATH] [--records N]

Description:
  Builds a tensor-proof dataset, forces Card 2 tensor-mode execution for top clustering,
  and verifies that cluster-stats reports:
    - kernel_backend_path == "cuda_tensor_fp16"
    - tensor_core_active == true

Notes:
  - Requires a built vectordb_v3_cli binary.
  - Intended for Linux/Ampere validation environments.
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
DATA_DIR="/tmp/vdb_card2_tensor_proof"
RECORDS="2048"

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
        --records)
            RECORDS="$2"
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

require_cmd python3

CLI="${BUILD_DIR}/vectordb_v3_cli"
if [[ ! -x "${CLI}" ]]; then
    CLI="${BUILD_DIR}/vectordb_v3_cli.exe"
fi
if [[ ! -x "${CLI}" ]]; then
    echo "error: vectordb_v3_cli not found under build dir: ${BUILD_DIR}" >&2
    echo "hint: build first with cmake --build ${BUILD_DIR} -j" >&2
    exit 2
fi

echo "== Card 2 Tensor Proof =="
echo "Repo root : ${REPO_ROOT}"
echo "Build dir : ${BUILD_DIR}"
echo "Data dir  : ${DATA_DIR}"
echo "Records   : ${RECORDS}"
echo

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

BULK_JSONL="${DATA_DIR}/tensor_proof_bulk.jsonl"
python3 - <<PY
import json
from pathlib import Path

records = int("${RECORDS}")
out = Path("${BULK_JSONL}")

rows = []
for i in range(records):
    cluster = i % 32
    base = 0.02 + cluster * 0.03
    vec = [round(base + ((d + i) % 17) * 0.0009, 6) for d in range(1024)]
    rows.append({"embedding_id": i + 1, "vector": vec})

out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {records} rows to {out}")
PY

echo "[1/4] init"
"${CLI}" init --path "${DATA_DIR}" >/dev/null

echo "[2/4] bulk-insert"
"${CLI}" bulk-insert --path "${DATA_DIR}" --input "${BULK_JSONL}" --batch-size 256 >/dev/null

echo "[3/4] build-top-clusters (forced tensor mode)"
TOP_JSONL="${DATA_DIR}/build_top_tensor_proof.jsonl"
env \
  VECTOR_DB_V3_KMEANS_BACKEND=cuda \
  VECTOR_DB_V3_KMEANS_PRECISION=tensor \
  VECTOR_DB_V3_FORCE_TENSOR_PATH=1 \
  VECTOR_DB_V3_TENSOR_MIN_OPS=1 \
  "${CLI}" build-top-clusters --path "${DATA_DIR}" --seed 7 > "${TOP_JSONL}"

echo "[4/4] cluster-stats verification"
STATS_JSON="${DATA_DIR}/cluster_stats_tensor_proof.json"
env \
  VECTOR_DB_V3_KMEANS_BACKEND=cuda \
  VECTOR_DB_V3_KMEANS_PRECISION=tensor \
  VECTOR_DB_V3_FORCE_TENSOR_PATH=1 \
  VECTOR_DB_V3_TENSOR_MIN_OPS=1 \
  "${CLI}" cluster-stats --path "${DATA_DIR}" > "${STATS_JSON}"

python3 - <<PY
import json
import sys
from pathlib import Path

top_events_path = Path("${TOP_JSONL}")
stage_end = None
for raw in top_events_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if isinstance(obj, dict) and obj.get("event_type") == "stage_end" and obj.get("stage_id") == "top":
        stage_end = obj
        break

if stage_end is None:
    print("ERROR: missing top stage_end event in build-top output", file=sys.stderr)
    sys.exit(1)

event_backend = stage_end.get("kernel_backend_path")
event_tensor_active = bool(stage_end.get("tensor_core_active"))
event_compliance = stage_end.get("compliance_status")
print(f"stage_end(top): backend={event_backend} tensor_active={event_tensor_active} compliance={event_compliance}")

path = Path("${STATS_JSON}")
stats = json.loads(path.read_text(encoding="utf-8"))
backend = stats.get("kernel_backend_path")
tensor_active = bool(stats.get("tensor_core_active"))
compliance = stats.get("compliance_status")

print(f"cluster-stats: backend={backend} tensor_active={tensor_active} compliance={compliance}")
print(f"stats file: {path}")
print(f"top stage events file: {top_events_path}")

if event_backend != "cuda_tensor_fp16":
    print("ERROR: top stage did not execute cuda_tensor_fp16", file=sys.stderr)
    sys.exit(1)
if not event_tensor_active:
    print("ERROR: top stage tensor_core_active is not true", file=sys.stderr)
    sys.exit(1)
if event_compliance != "pass":
    print("ERROR: top stage compliance_status is not pass", file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Tensor proof passed."
