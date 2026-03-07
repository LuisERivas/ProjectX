"""
Shared contract primitives used by both gateway and worker.

This subpackage holds:
- redis_keys: key builders for job hash, events stream, DLQ, and queue stream.
- serde: JSON serialization/deserialization policy.
- schema: shared validation helpers for queue messages/events.
- types: shared lightweight types/envelopes where useful.
- config: shared default stream/group names and tuning constants.
"""

