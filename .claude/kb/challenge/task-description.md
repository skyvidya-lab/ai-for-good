# Task Description — Track 1 Final Round

## Organisation

**ITU AI and Space Computing Challenge 2026** — Zhejiang Lab / Zero2x.
Track 1: **Space Intelligence Empowering Zero Hunger** (SDG 2).

## Input

For each test point:
- `point_id`, `Longitude`, `Latitude`
- Multiple `phenophase_date` rows (3–7 dates)
- Sentinel-2 L2A multi-band rasters at those dates

## Output

For each (point_id, phenophase_date) row, fill:
- `Pre_crop_type`: one of `{rice, corn, soybean, background}`
- `Pre_phenophase`: one of the 7 canonical phenophases

## Scoring

- **60% Algorithm Performance**: accuracy metrics on holdout test set (platform-evaluated).
- **40% Solution Design**: written document + presentation video.

## Deadlines

| Event | Date (UTC) |
|---|---|
| Validation environment open | 2026-04-06 |
| Validation environment closes | **2026-05-07** |
| Solution submission closes | **2026-05-10** |

## Submission Workflow

1. Login to `spaceaichallenge.zero2x.org`
2. Download data (done)
3. Local development
4. Model validation on platform
5. Project submission
6. Results display

## Journal Submission (Optional)

Participants can submit papers to **ITU J-FET Special Issue on AI and Space Computing**.
