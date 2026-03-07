"""
Worker-only contract implementation.

This subpackage provides:
- ContractClient: high-level worker API for the Redis job contract.
- Stream operations, runtime wiring, DLQ helpers, and atomic Lua-backed transitions.
- Lua scripts for atomic finalize+ack.
"""

