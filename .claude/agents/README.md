# Agent Routing — Dynamis Terra

> When to use which agent. Updated 2026-04-17.

See `_agentspec_README.md` for the upstream (AgentSpec) routing pattern we adopted.

---

## Agents by Category

### `ai-ml/` — Physics + Satellite Pipelines

| Agent | Use When |
|---|---|
| **dynamis-geo-architect** | Designing/changing Dynamis physics modules (MKM, Hurst, Innovation Loss, ChaosAttention). Kalman prior design. Loss formulations. |
| **satellite-data-engineer** | Sentinel-2 pipeline changes: TIFF parsing edge cases, rasterio CRS/resampling, folder consolidation, vegetation indices, NaN-safe extraction. |

### `domain/` — Agronomic + Submission

| Agent | Use When |
|---|---|
| **crop-classifier** | Interpreting model errors agronomically (rice vs corn vs soybean confusion). Feature engineering ideas rooted in phenology. OOD strategy for "background" class. |
| **submission-builder** | Preparing a platform submission CSV. Validating format, managing version numbers, packaging inference scripts. |

### `code-quality/` — Review + Clean Python

| Agent | Use When |
|---|---|
| **code-reviewer** | Review before committing significant new modules or refactors. Security / quality sweeps. |
| **python-developer** | Clean typed Python code. Dataclasses, generators, type hints. |

### `exploration/` — Search + Archaeology

| Agent | Use When |
|---|---|
| **codebase-explorer** | Large-codebase searches, prior-art discovery, tracing patterns across multiple workspaces. |

---

## SDD Workflow Agents (upstream from AgentSpec)

These are implicit in the 6 workflow commands under `.claude/commands/workflow/`. When you run `/brainstorm`, `/define`, `/design`, `/build`, `/ship`, `/iterate`, Claude Code uses the generic workflow agents described in `.claude/sdd/architecture/WORKFLOW_CONTRACTS.yaml`. They do not need separate agent files — the commands themselves carry the full process.

Note: AgentSpec's upstream has dedicated agent files (`brainstorm-agent.md` etc). We chose NOT to clone those because they reference 58 DE-specific agents we don't have. The commands work directly.

---

## Escalation / Matching Rules

| Task signal | Preferred agent |
|---|---|
| File path matches `src/dynamis/**/*.py` | `dynamis-geo-architect` |
| File path matches `src/data/**/*.py` | `satellite-data-engineer` |
| File path matches `src/models/*.py` | `dynamis-geo-architect` (model integration) |
| File path matches `src/submission/*.py` | `submission-builder` |
| Crop/phenophase domain question | `crop-classifier` |
| Review request (code diff, security) | `code-reviewer` |
| "Write me clean Python that does X" | `python-developer` |
| "Where is X in the codebase?" | `codebase-explorer` |

When two agents fit, prefer the more domain-specific (e.g. `dynamis-geo-architect` over `python-developer` for changes in `src/dynamis/`).

---

## Creating a New Agent

1. Copy `.claude/agents/_template.md` to `.claude/agents/{category}/{name}.md`.
2. Fill the frontmatter: `name`, `description` (with at least 2 `<example>` blocks), `tools`, `color`, `model`.
3. Add the routing row to this README.
4. Keep the agent focused — one responsibility, clear boundaries.
