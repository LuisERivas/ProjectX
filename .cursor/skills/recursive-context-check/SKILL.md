---
name: recursive-context-check
description: Recursively audits a target folder and all nested files/folders, compares each folder's contents to its local CONTEXT.md and CONTEXTCHECK.md where present, and writes a RECURSIVECONTEXTCHECK.md report with missing items, stale entries, duplicates, inaccuracies/contradictions, notes, recommended updates, and last-run timestamp.
---

# Recursive Context Check

## Purpose

Run a recursive documentation consistency audit for a target folder and produce one aggregate report file.

## Input

- A single target folder path.

## Scope Rules

- Include the target folder and all descendant folders/files recursively.
- For each folder encountered, inspect local `CONTEXT.md` and `CONTEXTCHECK.md` if present.
- Compare documented items against actual contents in that same folder.
- Do not use external folders as evidence.

## Required Workflow

1. Resolve the target folder path.
2. Recursively enumerate all folders/files beneath it.
3. For each folder:
   - Gather immediate contents (files and subfolders).
   - Read `CONTEXT.md` if present.
   - Read `CONTEXTCHECK.md` if present.
   - Identify:
     - Missing items (present on disk, missing from docs)
     - Stale entries (documented but missing on disk)
     - Duplicates/redundant entries
     - Inaccuracies/contradictions
4. Write one aggregate `RECURSIVECONTEXTCHECK.md` in the target folder root.

## Output File

Create or overwrite:

- `<target-folder>/RECURSIVECONTEXTCHECK.md`

Use this structure:

```markdown
# Recursive Context Check Report

- **Root folder:** `<path>`
- **Last checked:** `<YYYY-MM-DD HH:MM:SS timezone>`
- **Checker:** `recursive-context-check`

## Summary

- Folders scanned: `<n>`
- Files scanned: `<n>`
- Issues found: `<n>`
- Status: `PASS` or `NEEDS_UPDATE`

## Findings by Folder

### `<folder-path>`

#### Missing items
- `<item>` or `None`

#### Stale entries
- `<item>` or `None`

#### Duplicates
- `<item>` or `None`

#### Inaccuracies / Contradictions
- `<item>` or `None`

## Notes
- `<optional note>` or `None`

## Recommended updates
1. `<action>`
2. `<action>`
```

## Status Rules

- `PASS` if no issues are found across all scanned folders.
- `NEEDS_UPDATE` if any issue is found.

## Invocation Examples

- "Run recursive-context-check on `contract/`."
- "Audit `gateway/` recursively and write RECURSIVECONTEXTCHECK.md."
