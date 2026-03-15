from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        print("error: missing vectordb_v3_cli binary", file=sys.stderr)
        return 1

    data_dir = Path(tempfile.gettempdir()) / "vectordb_v3_scaffold_smoke"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    code, out, err = run([str(cli), "init", "--path", str(data_dir)], root)
    if code != 0:
        print(out)
        print(err, file=sys.stderr)
        return 1
    code, out, err = run([str(cli), "stats", "--path", str(data_dir)], root)
    if code != 0:
        print(out)
        print(err, file=sys.stderr)
        return 1
    code, out, err = run([str(cli), "unknown-cmd", "--path", str(data_dir)], root)
    if code != 2:
        print("error: unknown command should return exit code 2", file=sys.stderr)
        print(out)
        print(err, file=sys.stderr)
        return 1

    print("vectordb_v3_scaffold_smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
