from typing import Any, Dict

from contract.shared.serde import dumps, loads


JOB_HASH_JSON_FIELDS = {"payload", "result", "error"}


def serialize_job_hash(job_obj: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in job_obj.items():
        if k in JOB_HASH_JSON_FIELDS:
            out[k] = dumps(v)
        else:
            out[k] = "" if v is None else str(v)
    return out


def deserialize_job_hash(h: Dict[str, str]) -> Dict[str, Any]:
    job: Dict[str, Any] = dict(h)

    for k in JOB_HASH_JSON_FIELDS:
        raw = job.get(k)
        if raw is None or raw == "":
            job[k] = None if k in ("result", "error") else {}
            continue
        if isinstance(raw, str):
            try:
                job[k] = loads(raw)
            except Exception:
                job[k] = {"_raw": raw}

    for nk in ("created_ts", "updated_ts", "ttl_s"):
        if nk in job and isinstance(job[nk], str) and job[nk] != "":
            try:
                job[nk] = int(job[nk])
            except Exception:
                pass

    return job

