#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_card2_downstream_stage_proof.sh [--repo-root PATH] [--build-dir PATH] [--data-dir PATH] [--records N]

Description:
  Validates stages after Card 2 tensor scope by:
    1) preparing a synthetic dataset
    2) running top+mid with forced tensor mode
    3) running lower and final stages
    4) asserting lower/final stage_end compliance=pass and valid backend field presence
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
DATA_DIR="/tmp/vdb_card2_downstream_proof"
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

echo "== Card 2 Downstream Stage Proof =="
echo "Repo root : ${REPO_ROOT}"
echo "Build dir : ${BUILD_DIR}"
echo "Data dir  : ${DATA_DIR}"
echo "Records   : ${RECORDS}"
echo

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

BULK_JSONL="${DATA_DIR}/downstream_bulk.jsonl"
python3 - <<PY
import json
from pathlib import Path

records = int("${RECORDS}")
out = Path("${BULK_JSONL}")

rows = []
for i in range(records):
    cluster = i % 64
    base = 0.015 + cluster * 0.018
    vec = [round(base + ((d + i) % 23) * 0.0006, 6) for d in range(1024)]
    rows.append({"embedding_id": i + 1, "vector": vec})

out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {records} rows to {out}")
PY

echo "[1/6] init"
"${CLI}" init --path "${DATA_DIR}" >/dev/null

echo "[2/6] bulk-insert"
"${CLI}" bulk-insert --path "${DATA_DIR}" --input "${BULK_JSONL}" --batch-size 256 >/dev/null

echo "[3/6] build-top-clusters (forced tensor)"
env \
  VECTOR_DB_V3_KMEANS_BACKEND=cuda \
  VECTOR_DB_V3_KMEANS_PRECISION=tensor \
  VECTOR_DB_V3_FORCE_TENSOR_PATH=1 \
  VECTOR_DB_V3_TENSOR_MIN_OPS=1 \
  "${CLI}" build-top-clusters --path "${DATA_DIR}" --seed 7 > "${DATA_DIR}/downstream_top.jsonl"

echo "[4/6] build-mid-layer-clusters (forced tensor)"
env \
  VECTOR_DB_V3_KMEANS_BACKEND=cuda \
  VECTOR_DB_V3_KMEANS_PRECISION=tensor \
  VECTOR_DB_V3_FORCE_TENSOR_PATH=1 \
  VECTOR_DB_V3_TENSOR_MIN_OPS=1 \
  "${CLI}" build-mid-layer-clusters --path "${DATA_DIR}" --seed 7 > "${DATA_DIR}/downstream_mid.jsonl"

echo "[5/6] build-lower-layer-clusters"
LOWER_JSONL="${DATA_DIR}/downstream_lower.jsonl"
"${CLI}" build-lower-layer-clusters --path "${DATA_DIR}" --seed 7 > "${LOWER_JSONL}"

echo "[6/6] build-final-layer-clusters"
FINAL_JSONL="${DATA_DIR}/downstream_final.jsonl"
"${CLI}" build-final-layer-clusters --path "${DATA_DIR}" --seed 7 > "${FINAL_JSONL}"

python3 - <<PY
import json
import sys
from pathlib import Path

def parse_stage_end(path: Path, stage_id: str):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("event_type") == "stage_end" and obj.get("stage_id") == stage_id:
            return obj
    return None

lower_path = Path("${LOWER_JSONL}")
final_path = Path("${FINAL_JSONL}")
lower = parse_stage_end(lower_path, "lower")
final = parse_stage_end(final_path, "final")

if lower is None:
    print("ERROR: missing lower stage_end", file=sys.stderr)
    sys.exit(1)
if final is None:
    print("ERROR: missing final stage_end", file=sys.stderr)
    sys.exit(1)

print(f"stage_end(lower): backend={lower.get('kernel_backend_path')} tensor_active={lower.get('tensor_core_active')} compliance={lower.get('compliance_status')}")
print(f"stage_end(final): backend={final.get('kernel_backend_path')} tensor_active={final.get('tensor_core_active')} compliance={final.get('compliance_status')}")

for stage_name, event in [("lower", lower), ("final", final)]:
    if event.get("compliance_status") != "pass":
        print(f"ERROR: {stage_name} compliance_status is not pass", file=sys.stderr)
        sys.exit(1)
    if "kernel_backend_path" not in event:
        print(f"ERROR: {stage_name} missing kernel_backend_path", file=sys.stderr)
        sys.exit(1)
PY

echo
echo "Downstream stage proof passed."
