---
name: context-folder-audit-batch
description: Audits multiple specified folders (one folder per run, non-recursive), writes CONTEXTCHECK.md in each folder, then runs create-summary for each immediate file to enrich file summaries. Use when auditing more than one folder at once.
---

# Context Folder Audit Batch

## Purpose

Run a non-recursive folder audit workflow on a list of folders in one request, without relying on `CONTEXT.md`.

## Inputs

- A list of folder paths to audit.
- Optional summary output path (default: `ProjectX/CONTEXTCHECK_SUMMARY.md`).

## Required Workflow

1. Parse the folder list from the user request.
2. For each folder path:
   - Audit only immediate contents in that folder (single-folder, non-recursive).
   - Do not read, compare, or require `CONTEXT.md`.
   - Create or overwrite that folder's `CONTEXTCHECK.md` with a `## File Summaries` section and one placeholder entry per immediate file.
   - After writing `CONTEXTCHECK.md`, run the `create-summary` skill once per immediate file in that folder.
   - Each `create-summary` run must replace the target file entry with a 5-10 line summary that includes functionality and related files.
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
- Do not read, compare against, or mention `CONTEXT.md` as a source of truth.

## Per-folder output rule

For each folder, write `CONTEXTCHECK.md` as a self-contained folder audit report (no `CONTEXT.md` comparison), then enrich each file entry by running `create-summary` after the report is created.

Use this section at the end of each folder report:

```markdown
## File Summaries

- `<filename>`: `<1-sentence summary>`
- `<filename>`: `<1-3 sentence summary>`
```

File summary constraints:

- Include immediate files only (no recursive file summaries).
- Final summaries must follow `create-summary` output: 5-10 lines per file entry, including a related-files line.
- Use neutral, evidence-based wording based on the file's observed role/content.
- If a folder has no immediate files, write `- None`.

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

- Folder status is `PASS` when no discrepancies are found in the immediate folder audit.
- Folder status is `NEEDS_UPDATE` when one or more issues are found.

## Invocation examples

- "Run context-folder-audit-batch on `contract/`, `contract/shared/`, and `gateway/`."
- "Audit folders `[worker/, gateway/, contract/worker/]` and write a summary."
