# .claude/CLAUDE.md — Dynamis Terra Orchestration

**For:** Claude Code sessions working on this repository.
**Root context:** [../CLAUDE.md](../CLAUDE.md) (project overview + dataset anatomy).

---

## Mission

Deliver a physics-informed crop classifier (Dynamis) competitive on the 60% algorithm score AND a solution design document that dominates the 40% narrative score of the ITU AI and Space Computing Challenge 2026 Track 1 Final Round.

## Agents (by category)

| Agent | Category | When to use |
|---|---|---|
| `dynamis-geo-architect` | ai-ml | Designing physics-informed modules, Kalman priors, loss functions |
| `satellite-data-engineer` | ai-ml | Sentinel-2 parsing, rasterio pipelines, folder consolidation |
| `model-trainer` | ai-ml | Training loops, hyperparameter sweeps, GroupKFold logic |
| `crop-classifier` | domain | Agronomic reasoning (phenology, crop signatures, rice vs corn vs soybean) |
| `submission-builder` | domain | Platform submission format, output CSV packaging |
| `code-reviewer` | code-quality | Security/quality review before committing new modules |
| `python-developer` | code-quality | Clean dataclass/typed code patterns |
| `codebase-explorer` | exploration | Searching large codebases for prior art or patterns |

## Custom Commands

### Workflow (SDD 5-phase)
| Command | Purpose |
|---|---|
| `/brainstorm` | Phase 0 — explore ideas collaboratively |
| `/define` | Phase 1 — capture requirements + clarity score |
| `/design` | Phase 2 — architecture + file manifest |
| `/build` | Phase 3 — implement with agent delegation |
| `/ship` | Phase 4 — archive with lessons learned |
| `/iterate` | Cross-phase updates with cascade detection |

### Core utilities
| Command | Purpose |
|---|---|
| `/sync-context` | Regenerate auto-sections of CLAUDE.md from current repo state |
| `/memory` | Persist session insights to `.claude/storage/memory-{YYYY-MM-DD}.md` |
| `/readme-maker` | Generate README by analysing codebase |

### Training (ML)
| Command | Purpose |
|---|---|
| `/run-baseline` | Execute the LightGBM baseline pipeline |
| `/run-dynamis` | Execute the Dynamis classifier pipeline |

### Submission
| Command | Purpose |
|---|---|
| `/package-submission` | Format predictions for the platform |

## Knowledge Base Domains

| Domain | Path | Purpose |
|---|---|---|
| `dynamis` | `.claude/kb/dynamis/` | Core theory: MKM, Hurst, Innovation Loss, Chaos Attention |
| `sentinel2` | `.claude/kb/sentinel2/` | Bands, vegetation indices, L2A specifics |
| `crop-science` | `.claude/kb/crop-science/` | Phenology, crop signatures for rice/corn/soybean |
| `challenge` | `.claude/kb/challenge/` | Task description, dataset anatomy, submission format |

## Coding Standards

- **Python 3.10+**: use `from __future__ import annotations` + PEP 604 union types.
- **Type hints everywhere**, especially on public APIs.
- **No hidden global state.** Configuration via dataclasses (see `DynamisModelConfig`).
- **Rasterio operations must be NaN-safe** — satellite data always has missing pixels.
- **Seeds fixed** (`torch.manual_seed(42)`, `np.random.seed(42)`) in every training script.
- **Transfer Proof on every new training pipeline**: shuffle labels, confirm models drop to chance (~33% for 3 classes).

## Testing Strategy

- Local smoke tests in `tests/` use `background/.../test_input_sample/` (region03, 2 dates).
- Integration tests for pipeline + model sanity run against synthetic tensors.
- Colab notebook tests: run cells end-to-end in < 30 min on T4.

## SDD Workflow

Adopted from AgentSpec v3.0.0 on 2026-04-17. Every significant feature — Dynamis v3 physics injection, new sampling strategies, submission packaging changes — should flow through the 5-phase workflow:

```
/brainstorm "Feature idea"          # Phase 0: explore 2-3 approaches, YAGNI filter
/define <FEATURE_NAME>              # Phase 1: requirements + clarity score >=12/15
/design <FEATURE_NAME>               # Phase 2: architecture + file manifest + ADRs
/build <FEATURE_NAME>                # Phase 3: implement + tests + BUILD_REPORT
/ship <FEATURE_NAME>                 # Phase 4: archive + lessons learned
/iterate <file> "change description" # Cross-phase updates with cascade
```

Artefacts:
- In-flight → `.claude/sdd/features/{BRAINSTORM,DEFINE,DESIGN}_{FEATURE}.md`
- Reports → `.claude/sdd/reports/BUILD_REPORT_{FEATURE}.md`
- Shipped → `.claude/sdd/archive/{FEATURE}/SHIPPED_{YYYY-MM-DD}.md`

Templates and rules in `.claude/sdd/architecture/WORKFLOW_CONTRACTS.yaml`.

## Workflow Per Task Type

### Adding a new Dynamis module
1. Draft in `src/dynamis/`.
2. Update `src/dynamis/__init__.py` exports.
3. Add unit test in `tests/test_dynamis_modules.py`.
4. Document in `.claude/kb/dynamis/concepts/`.
5. For non-trivial additions, run the full SDD workflow (`/brainstorm` → `/ship`) so the design + lessons get archived.

### Adding a data pipeline step
1. Implement in `src/data/`.
2. Use `test_input_sample/region_test/` for smoke test.
3. Keep NaN-propagation explicit — never silently impute without logging.

### Modifying the model
1. Edit `src/models/dynamis_crop_classifier.py`.
2. Re-run Transfer Proof in the notebook.
3. Compare against baseline on same GroupKFold splits.

## Anti-Patterns

- Loading full 70 GB dataset for local dev. **Use sample regions** (5 × 4 folders ≈ 3 GB in `/content/sample/` on Colab).
- Treating the 4 folders as splits. **They are temporal batches** — always consolidate.
- Stratified K-fold without grouping. **Use GroupKFold by `point_id`** — the 7 phenophases of one point must stay in the same fold.
- Ignoring `mask` in the model. **Missing dates are real** — always propagate the `(B, T)` bool mask through attention.

## Reference Sources

- `C:\Users\eluzq\workspace\00_benchmarking\skyvidya_core\skyvidya_dynamis` — Dynamis generic core (HilbertEmbedding, MKM, HierarchicalHPR)
- `C:\Users\eluzq\workspace\dynamis-finance-ai` — PhysicsInformedAttention V4.2 (adapted)
- `C:\Users\eluzq\workspace\ai-powered-dynamis-nlp` — Transfer Proof protocol
- `C:\Users\eluzq\workspace\semana-ai-data-engineer-main` — `.claude/` orchestration template
