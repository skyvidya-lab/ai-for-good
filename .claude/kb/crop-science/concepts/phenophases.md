# The 7 Phenophases

The challenge uses a **MODIS-style 7-stage phenology**:

| Stage | Biological meaning | Typical window (NE China) |
|---|---|---|
| **Dormancy** | Pre-emergence / fallow soil | Winter → early Apr |
| **Greenup** | Initial leaf-out, rising NDVI | Late Apr – mid May |
| **MidGreenup** | Rapid canopy expansion | Late May – mid Jun |
| **Peak** | Maximum greenness (flowering/anthesis) | Jul |
| **Maturity** | Grain-fill; canopy plateau | Aug |
| **MidSenescence** | Yellowing; declining NDVI | Early Sep |
| **Senescence** | Near-harvest; canopy collapse | Mid-Sep → Oct |

## Ordinality

These 7 stages form a **near-cyclical state machine**:
`Dormancy → Greenup → MidGreenup → Peak → Maturity → MidSenescence → Senescence → Dormancy (next year)`

Only forward transitions and self-loops are physically valid. The phenology transition prior (see [phenology-transition-prior.md](../../dynamis/patterns/phenology-transition-prior.md)) encodes this.

## Observation semantics

Each labelled point has **one date per phenophase** (7 dates total per point). The challenge evaluates:
1. **Crop type** (rice / corn / soybean) — constant per point.
2. **Phenophase at each observation date** — the model must classify each of the 7 dates to the right phenophase.

## Key property of the canonical 7-stage

Equal representation: each phenophase appears in 778 training points (perfect balance). Makes class-weighted losses unnecessary for the phenophase head.
