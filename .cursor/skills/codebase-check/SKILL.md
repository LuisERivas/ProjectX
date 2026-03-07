---
name: codebase-check
description: Reviews the entire repository for conflicts, discrepancies, and duplicate logic/docs, then writes a root-level CODEBASECHECK.md report with timestamp, severity-ranked findings, and recommended fixes. Use when the user asks for a full codebase consistency audit.
---

# Codebase Check

## Purpose

Run a whole-repository audit for:
- Conflicts (behavioral or documentation contradictions)
- Discrepancies (code vs docs, runtime vs setup, stale references)
- Duplications (logic, config, architecture statements that can drift)

Then create a single report at project root:
- `CODEBASECHECK.md`

## Input

- No required input.  
- Optional user scope constraints (for example: "ignore docs" or "runtime only").

## Scope Rules

- Default scope is the entire repository recursively.
- Include active runtime code, tests, setup/deploy scripts, and docs.
- Exclude generated cache artifacts when evaluating findings (`__pycache__`, `.pyc` content).
- Treat `DEPRECATED` files as out of scope unless explicitly requested.
- Findings must be evidence-based (file path + concrete mismatch).

## Required Workflow

1. Enumerate repository files recursively.
2. Build the active architecture map from code:
   - gateway responsibilities
   - worker responsibilities
   - contract/shared responsibilities
   - setup/runtime env behavior
3. Compare for contradictions:
   - code vs code
   - code vs docs
   - setup scripts vs runtime expectations
   - tests vs claims in docs
4. Identify duplicates that can drift:
   - duplicated logic or constants
   - duplicated architecture guidance across docs
5. Classify findings by severity:
   - `HIGH`: correctness/reliability or data integrity risk
   - `MEDIUM`: architecture drift or operational risk
   - `LOW`: stale/misleading docs, minor duplication risk
6. Create or overwrite `CODEBASECHECK.md` at root with timestamp and results.

## Output File

Write to:
- `ProjectX/CODEBASECHECK.md`

Use this structure:

```markdown
# CODEBASE Check Report

- **Root folder:** `ProjectX/`
- **Last checked:** `<YYYY-MM-DD HH:MM:SS timezone>`
- **Checker:** `codebase-check`

## Summary

- Files scanned: `<n>`
- Findings: `<n>`
- High: `<n>`
- Medium: `<n>`
- Low: `<n>`
- Status: `PASS` or `NEEDS_UPDATE`

## Findings (By Severity)

### HIGH
- `<issue>`
  - **Paths:** `<path1>`, `<path2>`
  - **Why it matters:** `<impact>`
  - **Recommended fix:** `<action>`

### MEDIUM
- `<issue>`
  - **Paths:** `<path1>`, `<path2>`
  - **Why it matters:** `<impact>`
  - **Recommended fix:** `<action>`

### LOW
- `<issue>`
  - **Paths:** `<path1>`, `<path2>`
  - **Why it matters:** `<impact>`
  - **Recommended fix:** `<action>`

## Validation Gaps
- `<missing test or verification area>` or `None`

## Recommended Next Steps
1. `<highest-priority remediation>`
2. `<next remediation>`
3. `<verification step>`
```

## Status Rules

- `PASS` when no findings are detected.
- `NEEDS_UPDATE` when one or more findings exist.

## Quality Rules

- Do not report style preferences as findings.
- Do not claim a conflict without citing specific files.
- Prefer fewer high-confidence findings over broad vague lists.

## Invocation Examples

- "Run codebase-check."
- "Run codebase-check and ignore docs."
- "Audit the full repo for conflicts and write CODEBASECHECK.md."
