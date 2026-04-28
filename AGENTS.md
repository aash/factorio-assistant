# AGENTS Workflow Guide

This repository uses an agent-assisted workflow for maintaining `CHANGES.md` from git history.

## Purpose

- Keep a human-readable change log aligned with commit history.
- Standardize impact labeling and formatting.
- Make updates easy to review and append.

## Output File

- Primary file: `CHANGES.md`

## Required Format

1. Top title:
   - `# CHANGES`
2. Impact legend near top:
   - `Impact legend: 🔹 Low / 🔸 Med / 🔺 High`
3. One section per commit using a level-2 header:
   - `## <impact> <short-hash> / <YYYY-MM-DD> / <commit subject>`
4. Each section contains:
   - `**Explanation**`
   - Short bullet list (concise, 2–4 bullets preferred)
5. Section separator:
   - `---`
6. Do **not** include “Files changed” sections.
7. Do **not** include commits which sole reason to update CHANGES.md

## Impact Markers

- 🔹 Low: version bumps, lint/docs/minor maintenance
- 🔸 Med: additive features, non-breaking API/config/testing improvements
- 🔺 High: protocol/API semantic changes, synchronization/render correctness fixes, behavior changes likely to affect users

## Workflow Steps

1. Inspect history:
   - `git log --date=short --pretty=format:"%h|%ad|%s"`
2. Draft concise explanations per commit.
3. Assign impact marker per commit.
4. Write/update `CHANGES.md` in the required format.
5. Validate consistency:
   - All commits represented in chosen range
   - Header format is consistent
   - Explanation bullets are short and clear

## Incremental Update Mode (Recommended)

When adding new commits only:

1. Read current top section in `CHANGES.md`.
2. Collect commits newer than that hash.
3. Prepend new sections at the top (newest first), preserving existing content below.


## Style Rules

- Keep wording factual and concise.
- Prefer behavior/impact language over implementation trivia.
- Avoid speculative statements.
- Keep commit subjects as-is in headers; normalize only explanation text.
