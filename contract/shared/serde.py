import json
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None


def dumps(obj: Any) -> str:
    if orjson is not None:
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def loads(s: str) -> Any:
    if orjson is not None:
        return orjson.loads(s)
    return json.loads(s)

