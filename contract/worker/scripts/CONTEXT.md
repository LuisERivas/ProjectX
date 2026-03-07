# Location: `ProjectX/contract/worker/scripts/`

Scope: Immediate contents only (non-recursive).

This folder contains:

- `finalize_and_ack.lua`: Lua script that performs atomic terminal job finalization by updating job hash fields, appending terminal event data, applying TTL, and acknowledging the queue stream message.
- `mark_running.lua`: Lua script that atomically transitions `queued -> running`, appends `running` event data, applies TTL, and ACKs non-runnable deliveries.
- `CONTEXT.md`: folder-level inventory for worker contract scripts.
- `CONTEXTCHECK.md`: most recent folder context audit report.

