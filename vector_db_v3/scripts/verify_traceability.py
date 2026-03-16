from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROW_RE = re.compile(r"^\|\s*(?P<req>.+?)\s*\|\s*(?P<contracts>.+?)\s*\|\s*(?P<gates>.+?)\s*\|\s*$")
GATE_RE = re.compile(r"\bG[1-7]\b")


def parse_matrix(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        req = m.group("req").strip()
        if req.lower() == "requirement area" or req.startswith("---"):
            continue
        rows.append(
            {
                "requirement": req,
                "contracts": [c.strip().strip("`") for c in m.group("contracts").split(",") if c.strip()],
                "gates": GATE_RE.findall(m.group("gates")),
            }
        )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify TRACEABILITY_MATRIX.md and gate map consistency.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--gate-map", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    gate_map_path = Path(args.gate_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not matrix_path.exists():
        print(f"error: matrix not found: {matrix_path}", file=sys.stderr)
        return 2
    if not gate_map_path.exists():
        print(f"error: gate map not found: {gate_map_path}", file=sys.stderr)
        return 2

    gate_map = json.loads(gate_map_path.read_text(encoding="utf-8"))
    rows = parse_matrix(matrix_path)

    missing: list[str] = []
    warnings: list[str] = []

    active_gates = set(gate_map.keys())
    matrix_gates = set()
    for row in rows:
        for gate in row["gates"]:
            matrix_gates.add(gate)
            if gate not in active_gates:
                warnings.append(f"matrix references gate not active in this run: {gate} ({row['requirement']})")

    for gate_id, gate_def in gate_map.items():
        refs = gate_def.get("contract_refs", [])
        if not refs:
            missing.append(f"{gate_id}: missing contract_refs in gate_map.json")
        if gate_id not in matrix_gates:
            warnings.append(f"{gate_id}: gate active but not referenced by TRACEABILITY_MATRIX.md")

    # Ensure matrix rows that mention active gates also carry a contract mapping.
    for row in rows:
        active_row_gates = [g for g in row["gates"] if g in active_gates]
        if active_row_gates and not row["contracts"]:
            missing.append(f"row '{row['requirement']}' has active gates {active_row_gates} but no contracts")

    ok = len(missing) == 0
    result = {
        "status": "pass" if ok else "fail",
        "matrix_path": str(matrix_path),
        "gate_map_path": str(gate_map_path),
        "missing_links": missing,
        "warnings": warnings,
        "active_gates": sorted(active_gates),
    }
    write_json(out_dir / "result.json", result)
    (out_dir / "missing_links.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

