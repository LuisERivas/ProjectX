def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def events_key(job_id: str) -> str:
    return f"job:{job_id}:events"


def dlq_stream_key() -> str:
    return "jobs:dlq"


def communications_text_key() -> str:
    return "communications:text"

