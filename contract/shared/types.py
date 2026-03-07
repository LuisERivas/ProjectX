from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class QueueMessage:
    msg_id: str
    fields: Dict[str, str]


@dataclass
class JobEnvelope:
    job_id: str
    msg_id: str
    task: str
    payload_raw: str
    ttl_s: int
    fields: Dict[str, Any]

