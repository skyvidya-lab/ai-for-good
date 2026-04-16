---
name: sync-context
description: Regenerate auto-sections of CLAUDE.md from current repo state (structure, modules, KB inventory).
---

# /sync-context

Analyses the repo and refreshes sections of `CLAUDE.md` that drift as the code evolves.

## Usage

```
/sync-context            # Full refresh of auto sections
/sync-context --dry-run  # Preview only
```

## Sections Managed

| Section | Mode | Source |
|---|---|---|
| Project Overview | Preserve | Manual |
| Dataset Anatomy | Preserve | Manual (changes only if `analise-dataset.md` changes) |
| Dynamis Component Map | Refresh | Scan `src/dynamis/*.py` |
| Repository Structure | Refresh | `tree` of repo |
| Quickstart | Preserve | Manual |
| Conventions | Preserve | Manual |

## Workflow

1. Read existing `CLAUDE.md`.
2. Glob `src/**/*.py` for module inventory.
3. Glob `.claude/kb/**/*.md` for KB inventory.
4. Glob `notebooks/*.ipynb` for notebook list.
5. Diff against current `CLAUDE.md` and apply only the refresh-mode sections.
6. Show the diff before saving.
