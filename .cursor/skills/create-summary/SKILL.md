---
name: create-summary
description: Reviews one target file and updates the same folder's CONTEXTCHECK.md (non-recursive) by replacing that file's summary with a 5-10 line summary covering functionality and related files. Use when the user asks to improve or rewrite a file summary in CONTEXTCHECK.md.
---

# Create Summary

## Purpose

Update one file's summary inside the local `CONTEXTCHECK.md` so it is more complete and actionable.

## Inputs

- Target file path (required).
- Optional emphasis areas (for example: API behavior, data flow, edge cases).

## Scope Rules (Strict)

- Non-recursive: operate only in the target file's immediate folder.
- Edit only that folder's `CONTEXTCHECK.md`.
- Replace only the target file's summary entry.
- Do not modify summaries for other files unless the user explicitly asks.

## Required Workflow

1. Identify the target file and its immediate parent folder.
2. Read the target file to understand:
   - primary functionality
   - key inputs/outputs
   - major responsibilities and side effects
3. Identify related files and include only concrete relationships, such as:
   - imports used by the target file
   - files that import/call the target file's public symbols
   - neighboring files in the same folder that the target file directly coordinates with
4. Open `<parent-folder>/CONTEXTCHECK.md`.
5. Locate the target file entry under `## File Summaries`.
6. Replace that single entry with a 5-10 line summary block in this format:

```markdown
- `<filename>`:
  - `<line 1: primary purpose/functionality>`
  - `<line 2: key runtime behavior>`
  - `<line 3: important inputs/outputs or interfaces>`
  - `<line 4: error handling / edge-case behavior>`
  - `<line 5: related files: file_a, file_b>`
  - `<optional lines 6-10 with additional concrete details>`
```

7. If the target file entry does not exist, add it under `## File Summaries` using the same 5-10 line format.
8. Keep statements evidence-based and specific; avoid vague wording.

## Related Files Requirement

- Include a `related files` line in every summary.
- List file paths relative to the same folder when possible.
- If no strong relationship can be verified, write: `related files: none identified`.

## Output Expectations

- Preserve existing `CONTEXTCHECK.md` structure and headings.
- Keep summary length between 5 and 10 lines total for the target entry.
- Use concise, factual language.

## Invocation Examples

- "Use create-summary for `gateway/main.py`."
- "Update the summary for `contract/gateway/service.py` in its `CONTEXTCHECK.md`."
