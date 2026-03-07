import os
import subprocess
import sys
from pathlib import Path


def run(cmd, *, check=True):
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=check)


def ensure_redis():
    print("\n=== Installing and enabling Redis (requires sudo) ===")
    run(["sudo", "apt", "update"])
    run(["sudo", "apt", "install", "-y", "redis-server"])
    run(["sudo", "systemctl", "enable", "redis-server"])
    run(["sudo", "systemctl", "start", "redis-server"])


def ensure_venv(project_root: Path):
    print("\n=== Creating Python virtualenv and installing dependencies ===")
    venv_dir = project_root / ".venv"
    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
    pip = venv_dir / "bin" / "pip"
    if not pip.exists():
        # Windows-style path fallback (in case this is run elsewhere)
        pip = venv_dir / "Scripts" / "pip.exe"
    run([str(pip), "install", "--upgrade", "pip"])
    run(
        [
            str(pip),
            "install",
            "fastapi",
            "uvicorn",
            "redis==7.2.1",
            "pydantic",
            "pytest",
        ]
    )
    return venv_dir


def write_systemd_units(project_root: Path, venv_dir: Path):
    print("\n=== Writing systemd unit files (requires sudo) ===")
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or "jetson"

    project_workdir = project_root

    # Choose the Unix-style venv python binary; on Jetson this is correct.
    python_bin = venv_dir / "bin" / "python"

    gateway_unit = f"""[Unit]
Description=ProjectX Redis HTTP gateway
After=network-online.target redis-server.service
Wants=network-online.target

[Service]
User={user}
Group={user}
WorkingDirectory={project_workdir}
Environment=PYTHONPATH={project_root}
Environment=REDIS_URL=redis://127.0.0.1:6379/0
Environment=QUEUE_STREAM_KEY=jobs:stream
Environment=WORKER_GROUP=workers
Environment=JOB_TTL_S=3600
Environment=BACKPRESSURE_MAX_BACKLOG=200
Environment=SSE_BLOCK_MS=15000
Environment=SSE_HEARTBEAT_S=15
ExecStart={python_bin} -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""

    worker_unit = f"""[Unit]
Description=ProjectX Redis echo worker
After=network-online.target redis-server.service
Wants=network-online.target

[Service]
User={user}
Group={user}
WorkingDirectory={project_workdir}
Environment=PYTHONPATH={project_root}
Environment=REDIS_URL=redis://127.0.0.1:6379/0
Environment=QUEUE_STREAM_KEY=jobs:stream
Environment=WORKER_GROUP=workers
Environment=DEFAULT_TTL_S=3600
Environment=BLOCK_MS=5000
Environment=COUNT=10
Environment=MAX_INFLIGHT=4
Environment=JOB_TIMEOUT_S=0
ExecStart={python_bin} -m worker.worker_main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""

    tmp_gateway = project_root / "redis-gateway.service"
    tmp_worker = project_root / "redis-echo-worker.service"
    tmp_gateway.write_text(gateway_unit, encoding="utf-8")
    tmp_worker.write_text(worker_unit, encoding="utf-8")

    run(
        [
            "sudo",
            "cp",
            str(tmp_gateway),
            "/etc/systemd/system/redis-gateway.service",
        ]
    )
    run(
        [
            "sudo",
            "cp",
            str(tmp_worker),
            "/etc/systemd/system/redis-echo-worker.service",
        ]
    )
    run(["sudo", "systemctl", "daemon-reload"])
    run(["sudo", "systemctl", "enable", "redis-gateway.service"])
    run(["sudo", "systemctl", "enable", "redis-echo-worker.service"])
    run(["sudo", "systemctl", "start", "redis-gateway.service"])
    run(["sudo", "systemctl", "start", "redis-echo-worker.service"])


def main():
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")

    ensure_redis()
    venv_dir = ensure_venv(project_root)
    write_systemd_units(project_root, venv_dir)

    print("\nSetup complete.")
    print("You can check services with:")
    print("  sudo systemctl status redis-gateway.service")
    print("  sudo systemctl status redis-echo-worker.service")


if __name__ == "__main__":
    main()

