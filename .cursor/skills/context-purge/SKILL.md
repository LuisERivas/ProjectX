---
name: context-purge
description: Renames all CONTEXT.md and CONTEXTCHECK.md files across the repository to deprecated names. Use when the user asks to deprecate, retire, or bulk-rename context documentation files.
---

# Context Purge

## Purpose

Deprecate context documentation filenames across the whole repo by renaming:

- `CONTEXT.md` -> `CONTEXTDEPRECATED.md`
- `CONTEXTCHECK.md` -> `CONTEXTCHECKDEPRECATED.md`

## Instructions

1. Search the full repository for both filename patterns.
2. Build two rename sets:
   - every `CONTEXT.md` file
   - every `CONTEXTCHECK.md` file
3. For each file, rename in-place within the same folder to the deprecated target name.
4. If the destination file already exists in a folder, do not overwrite it. Skip that file and report it as a conflict.
5. After renaming, verify both original filename patterns are gone (or only the skipped conflict cases remain).
6. Report:
   - number of files renamed for each pattern
   - any skipped conflicts
   - final verification result

## Output Format

Use this structure in your final response:

```markdown
Renamed context files to deprecated names across the repository.

- `CONTEXT.md` -> `CONTEXTDEPRECATED.md`: <count>
- `CONTEXTCHECK.md` -> `CONTEXTCHECKDEPRECATED.md`: <count>
- Conflicts skipped: <count or none>
- Verification: <pass/fail with short reason>
```

## Notes

- Apply this recursively for the entire repository unless the user asks for a narrower scope.
- Keep directory structure unchanged; only rename filenames.
- Never delete file contents as part of this operation.
