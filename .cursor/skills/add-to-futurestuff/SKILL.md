---
name: add-to-futurestuff
description: Adds a new task entry to ProjectX root ignoreFUTURESTUFF.md. Use when the user asks to save something for later, add future work, backlog an item, or append a task to ignoreFUTURESTUFF.md.
---

# Add To Futurestuff

## Purpose

Append user-requested future tasks to `ignoreFUTURESTUFF.md` in the project root in a consistent format.

## Required Workflow

1. Get the exact task text from the user request.
2. Target file path is always `ProjectX/ignoreFUTURESTUFF.md`.
3. If the file does not exist, create it with:
   - `# ignoreFUTURESTUFF`
   - blank line
   - `## Tasks`
4. Ensure there is a `## Tasks` section. If missing, add it at the end.
5. Add the task as an unchecked checklist line:
   - `- [ ] <task text>`
6. Avoid duplicate task lines (case-insensitive exact text match after trimming).
7. Keep existing content intact; do not reorder or rewrite unrelated lines.

## Format Rules

- Use plain markdown checklist entries only.
- One task per line.
- Preserve user wording unless tiny cleanup is needed for clarity.
- Do not add timestamps unless user explicitly requests them.

## Success Criteria

- `ignoreFUTURESTUFF.md` exists at project root.
- New task appears once under `## Tasks`.
- No unrelated content changes were made.

## Invocation Examples

- "Add this to future stuff: move retries into shared helper."
- "Put this in ignoreFUTURESTUFF.md: evaluate websocket streaming."
- "Backlog this task for later: add timeout integration tests."

