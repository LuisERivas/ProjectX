from typing import Dict

from contract.shared.errors import SchemaError


def validate_queue_message(fields: Dict[str, str]) -> None:
    if "job_id" not in fields or "task" not in fields or "payload" not in fields:
        raise SchemaError("queue message missing required fields")

