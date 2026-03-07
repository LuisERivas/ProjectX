---
name: context-folder-audit
description: Audits files in one specified folder only (non-recursive), compares them against that same folder's CONTEXT.md, and writes CONTEXTCHECK.md with discrepancies, gaps, duplications, and the last-check timestamp. Use when validating single-folder documentation accuracy.
---

# Context Folder Audit

## Purpose

Validate whether a target folder's `CONTEXT.md` accurately describes what exists in that same folder (non-recursive).

## Required Workflow

1. Get the target folder path from the user.
2. Restrict analysis strictly to that exact folder path (non-recursive).
3. Enumerate only immediate files and immediate subfolder names in that folder (do not read inside subfolders unless explicitly asked in a separate run).
4. Read `CONTEXT.md` in the target folder root.
5. Compare `CONTEXT.md` statements against actual in-scope files.
6. Identify:
   - Missing from context (exists on disk, not documented)
   - Stale context entries (documented, no longer exists)
   - Duplications/redundant statements
   - Contradictions/inaccuracies
   - Ambiguous wording that can mislead
7. Create or overwrite `CONTEXTCHECK.md` in that same target folder.

## Scope Rules (Strict)

- Use only files from the target folder itself for findings.
- Do not use external files as evidence.
- Do not scan descendant folders.
- Do not compare against neighboring folders.
- If `CONTEXT.md` is missing, still generate `CONTEXTCHECK.md` with status `NEEDS_UPDATE`.

## Output File Requirements

Write `CONTEXTCHECK.md` in this exact structure:

```markdown
# CONTEXT Check Report

- **Folder:** `<path>`
- **Last checked:** `<YYYY-MM-DD HH:MM:SS timezone>`
- **Checker:** `context-folder-audit`

## Summary

- Files scanned: `<n>`
- Issues found: `<n>`
- Status: `PASS` or `NEEDS_UPDATE`

## Findings

### Missing from CONTEXT.md
- `<item>` or `None`

### Stale entries in CONTEXT.md
- `<item>` or `None`

### Duplications
- `<item>` or `None`

### Inaccuracies / Contradictions
- `<item>` or `None`

### Notes
- `<optional note>` or `None`

## Recommended updates
1. `<action>`
2. `<action>`
```

## Result Rules

- If no issues are found, set status to `PASS` and explicitly state no discrepancies.
- If any issue exists, set status to `NEEDS_UPDATE`.
- Keep findings concrete and file-path specific where possible.

## Invocation Examples

- "Audit `contract/shared` context."
- "Run context-folder-audit on `gateway/`."
- "Check `worker/` and write CONTEXTCHECK.md."
