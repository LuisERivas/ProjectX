---
name: searcher
description: Searches codebase content from user-provided criteria and executes a requested follow-up action. Use when the user asks to find references, scan files for patterns, gather file+line evidence, or generate a markdown findings document from search results.
---
# Searcher

## Purpose

Take a **search criteria** and an **action**, run the search, and complete the action with clear, auditable outputs.

## Inputs

- `criteria`: What to search for (keywords, phrases, regex-like patterns, scope).
- `action`: What to do with results (report, summarize, update document, checklist, etc.).
- Optional:
  - `scope`: folder(s), file globs, include/exclude hints.
  - `output_path`: where to write report if requested.

If inputs are missing, ask only for the minimum required clarification.

## Default Behavior

When no output format is specified, produce a **Markdown report** with:
- Search criteria used
- Scope searched
- Match count summary
- Findings with:
  - file path
  - line number
  - short matched snippet
- Action result section

## Workflow

1. Parse request into `criteria`, `action`, and `scope`.
2. Search using `rg` (prefer precise patterns and scoped paths).
3. If matches are large/noisy, refine query and rerun.
4. Gather evidence (file path + line number + concise snippet).
5. Execute requested action:
   - If asked to create a doc, write a markdown file.
   - If asked to summarize only, return concise findings in chat.
6. Validate that every claim in output is backed by a concrete match.

## Output Template (Default Markdown Report)

```markdown
# Search Report

## Criteria
- "<criteria>"

## Scope
- <folder_or_glob>

## Summary
- Total files matched: <n>
- Total matches: <m>

## Findings
- `<path>`:<line>
  - `<snippet>`
- `<path>`:<line>
  - `<snippet>`

## Action Result
- <what was produced or changed>
```

## Examples

- "Search through all files in `vector_db_v2` for metadata filters and create a document with file names and line numbers."
- "Find references to `build-second-level-clusters` and summarize where it is called."
- "Scan `vector_db/src` for `k_min` and list all write/read points."

## Guardrails

- Prefer `rg` over broad manual scanning.
- Keep snippets short and relevant.
- Do not invent matches; only report verified results.
- If no matches are found, state that explicitly and include attempted criteria/scope.
