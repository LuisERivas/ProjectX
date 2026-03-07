---
name: context-folder-audit-batch
description: Runs context-folder-audit across multiple specified folders (one folder per run, non-recursive) and writes CONTEXTCHECK.md in each folder, with an optional aggregate summary file. Use when auditing more than one folder at once.
---

# Context Folder Audit Batch

## Purpose

Run the single-folder `context-folder-audit` workflow on a list of folders in one request.

## Inputs

- A list of folder paths to audit.
- Optional summary output path (default: `ProjectX/CONTEXTCHECK_SUMMARY.md`).

## Required Workflow

1. Parse the folder list from the user request.
2. For each folder path:
   - Run the same logic as `context-folder-audit` (single-folder, non-recursive).
   - Create or overwrite that folder's `CONTEXTCHECK.md`.
3. After all folders are processed, create/update aggregate summary file with:
   - run timestamp
   - folders processed
   - pass/fail status per folder
   - issue count per folder
   - total issues

## Per-folder scope rules (strict)

- Audit exactly the specified folder only (non-recursive).
- Use only immediate files in that folder for findings.
- Do not inspect descendant folders unless they are explicitly listed as separate targets.
- Do not use external files as evidence.

## Per-folder output rule

For each folder, write `CONTEXTCHECK.md` using the format defined by `context-folder-audit`.

## Aggregate summary format

Write summary markdown in this structure:

```markdown
# Context Audit Batch Summary

- **Last run:** `<YYYY-MM-DD HH:MM:SS timezone>`
- **Checker:** `context-folder-audit-batch`
- **Folders processed:** `<n>`
- **Total issues:** `<n>`

## Folder Results

| Folder | Status | Issues | Report |
|---|---:|---:|---|
| `path/a` | `PASS` | 0 | `path/a/CONTEXTCHECK.md` |
| `path/b` | `NEEDS_UPDATE` | 2 | `path/b/CONTEXTCHECK.md` |
```

## Status rules

- Folder status is `PASS` when no discrepancies are found.
- Folder status is `NEEDS_UPDATE` when one or more issues are found.

## Invocation examples

- "Run context-folder-audit-batch on `contract/`, `contract/shared/`, and `gateway/`."
- "Audit folders `[worker/, gateway/, contract/worker/]` and write a summary."
