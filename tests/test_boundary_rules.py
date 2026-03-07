from pathlib import Path
import re


FORBIDDEN_PATTERNS = [
    r"\.xadd\(",
    r"\.xack\(",
    r"\.hset\(",
    r"\.hgetall\(",
    r"\.expire\(",
    r"\.xreadgroup\(",
    r"\.xautoclaim\(",
    r"\.xgroup_create\(",
]


def _assert_no_forbidden_redis_calls(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in FORBIDDEN_PATTERNS:
        assert re.search(pattern, text) is None, f"Forbidden Redis call in {path}: {pattern}"


def test_worker_main_no_direct_redis_data_plane_calls() -> None:
    _assert_no_forbidden_redis_calls(Path("worker/worker_main.py"))


def test_gateway_main_no_direct_redis_data_plane_calls() -> None:
    _assert_no_forbidden_redis_calls(Path("gateway/main.py"))

