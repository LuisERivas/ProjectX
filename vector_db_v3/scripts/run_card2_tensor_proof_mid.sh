#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_card2_tensor_proof_mid.sh [--repo-root PATH] [--build-dir PATH] [--data-dir PATH] [--records N]

Description:
  Proves Card 2 tensor execution for MID stage by:
    1) building a synthetic dataset
    2) running top stage (prerequisite)
    3) forcing tensor mode on mid stage
    4) validating stage_end(mid) reports:
       - kernel_backend_path == cuda_tensor_fp16
       - tensor_core_active == true
       - compliance_status == pass
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
DATA_DIR="/tmp/vdb_card2_tensor_proof_mid"
RECORDS="4096"

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
    exit 2
fi

echo "== Card 2 Tensor Proof (Mid) =="
echo "Repo root : ${REPO_ROOT}"
echo "Build dir : ${BUILD_DIR}"
echo "Data dir  : ${DATA_DIR}"
echo "Records   : ${RECORDS}"
echo

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

BULK_JSONL="${DATA_DIR}/tensor_proof_mid_bulk.jsonl"
python3 - <<PY
import json
from pathlib import Path

records = int("${RECORDS}")
out = Path("${BULK_JSONL}")

rows = []
for i in range(records):
    cluster = i % 64
    base = 0.01 + cluster * 0.02
    vec = [round(base + ((d + i) % 19) * 0.0008, 6) for d in range(1024)]
    rows.append({"embedding_id": i + 1, "vector": vec})

out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {records} rows to {out}")
PY

echo "[1/5] init"
"${CLI}" init --path "${DATA_DIR}" >/dev/null

echo "[2/5] bulk-insert"
"${CLI}" bulk-insert --path "${DATA_DIR}" --input "${BULK_JSONL}" --batch-size 256 >/dev/null

echo "[3/5] build-top-clusters (prereq)"
env \
  VECTOR_DB_V3_KMEANS_BACKEND=cuda \
  VECTOR_DB_V3_KMEANS_PRECISION=tensor \
  VECTOR_DB_V3_FORCE_TENSOR_PATH=1 \
  VECTOR_DB_V3_TENSOR_MIN_OPS=1 \
  "${CLI}" build-top-clusters --path "${DATA_DIR}" --seed 7 > "${DATA_DIR}/build_top_midproof.jsonl"

echo "[4/5] build-mid-layer-clusters (forced tensor mode)"
MID_JSONL="${DATA_DIR}/build_mid_tensor_proof.jsonl"
env \
  VECTOR_DB_V3_KMEANS_BACKEND=cuda \
  VECTOR_DB_V3_KMEANS_PRECISION=tensor \
  VECTOR_DB_V3_FORCE_TENSOR_PATH=1 \
  VECTOR_DB_V3_TENSOR_MIN_OPS=1 \
  "${CLI}" build-mid-layer-clusters --path "${DATA_DIR}" --seed 7 > "${MID_JSONL}"

echo "[5/5] validate stage_end(mid)"
python3 - <<PY
import json
import sys
from pathlib import Path

path = Path("${MID_JSONL}")
stage_end = None
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if isinstance(obj, dict) and obj.get("event_type") == "stage_end" and obj.get("stage_id") == "mid":
        stage_end = obj
        break

if stage_end is None:
    print("ERROR: missing stage_end(mid)", file=sys.stderr)
    sys.exit(1)

backend = stage_end.get("kernel_backend_path")
tensor_active = bool(stage_end.get("tensor_core_active"))
compliance = stage_end.get("compliance_status")
print(f"stage_end(mid): backend={backend} tensor_active={tensor_active} compliance={compliance}")
print(f"events file: {path}")

if backend != "cuda_tensor_fp16":
    print("ERROR: mid stage did not execute cuda_tensor_fp16", file=sys.stderr)
    sys.exit(1)
if not tensor_active:
    print("ERROR: mid stage tensor_core_active is not true", file=sys.stderr)
    sys.exit(1)
if compliance != "pass":
    print("ERROR: mid stage compliance_status is not pass", file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Mid-stage tensor proof passed."
