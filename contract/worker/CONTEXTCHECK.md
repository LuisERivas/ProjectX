# CONTEXTCHECK

- **Folder:** `contract/worker`
- **Last run:** `2026-03-10 15:52:42 Pacific Daylight Time`
- **Checker:** `context-folder-audit-batch`
- **Scope:** `non-recursive immediate files only`
- **CONTEXT.md dependency:** `disabled`

## Findings
- Reviewed `7` immediate files in this folder.
- No discrepancies detected under the non-recursive folder-only audit criteria.

## Status
- **Result:** `PASS`
- **Issue count:** `0`

## File Summaries
- `CONTEXTCHECK.md`:
  - Markdown documentation/report file maintained in `contract/worker`.
  - Provides human-readable operational context rather than executable logic.
  - Captures run metadata, findings, and references for maintainers.
  - Accuracy depends on synchronization with current code and folder contents.
  - related files: none identified
  - Observed size is 972 bytes across 24 lines (20 non-empty lines).

- `__init__.py`:
  - Python module in `contract/worker` with 0 classes and 0 functions.
  - Contains executable logic used by this package and its call sites.
  - Defines symbols consumed through imports and module-level interfaces.
  - Edge-case behavior depends on explicit checks and raised exceptions in code paths.
  - related files: contract_client.py, dlq.py
  - Observed size is 265 bytes across 9 lines (7 non-empty lines).

- `contract_client.py`:
  - Python module in `contract/worker` with 1 classes and 3 functions.
  - Contains executable logic used by this package and its call sites.
  - Defines symbols consumed through imports and module-level interfaces.
  - Edge-case behavior depends on explicit checks and raised exceptions in code paths.
  - related files: __init__.py, dlq.py, errors.py, stream_ops.py
  - Observed size is 7220 bytes across 221 lines (197 non-empty lines).

- `dlq.py`:
  - Python module in `contract/worker` with 0 classes and 0 functions.
  - Contains executable logic used by this package and its call sites.
  - Defines symbols consumed through imports and module-level interfaces.
  - Edge-case behavior depends on explicit checks and raised exceptions in code paths.
  - related files: __init__.py, contract_client.py
  - Observed size is 264 bytes across 11 lines (6 non-empty lines).

- `errors.py`:
  - Python module in `contract/worker` with 0 classes and 0 functions.
  - Contains executable logic used by this package and its call sites.
  - Defines symbols consumed through imports and module-level interfaces.
  - Edge-case behavior depends on explicit checks and raised exceptions in code paths.
  - related files: __init__.py, contract_client.py
  - Observed size is 140 bytes across 7 lines (6 non-empty lines).

- `runtime.py`:
  - Python module in `contract/worker` with 1 classes and 1 functions.
  - Contains executable logic used by this package and its call sites.
  - Defines symbols consumed through imports and module-level interfaces.
  - Edge-case behavior depends on explicit checks and raised exceptions in code paths.
  - related files: __init__.py, contract_client.py
  - Observed size is 1302 bytes across 42 lines (34 non-empty lines).

- `stream_ops.py`:
  - Python module in `contract/worker` with 0 classes and 0 functions.
  - Contains executable logic used by this package and its call sites.
  - Defines symbols consumed through imports and module-level interfaces.
  - Edge-case behavior depends on explicit checks and raised exceptions in code paths.
  - related files: __init__.py, contract_client.py
  - Observed size is 1855 bytes across 70 lines (61 non-empty lines).
