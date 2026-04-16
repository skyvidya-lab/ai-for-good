---
name: package-submission
description: Format predictions for the challenge platform.
---

# /package-submission

Converts model predictions into the platform-required CSV format.

## Usage

```
/package-submission --model models/dynamis_v1.pt --out submissions/dynamis_v1.csv
/package-submission --ensemble models/dynamis_v1.pt models/dynamis_v1b.pt
```

## Process

1. Load test points from `test_point.csv`.
2. Build per-point PointSeries from consolidated data.
3. Run inference (average over ensemble if requested).
4. Apply uncertainty threshold → route high-uncertainty points to `background`.
5. Write CSV matching input row order exactly.
6. Validate structure (columns, dates, no empty cells).

## Validation Checklist

- [ ] Column order matches `test_point.csv`.
- [ ] Row count matches.
- [ ] All `Pre_crop_type` ∈ {rice, corn, soybean, background}.
- [ ] All `Pre_phenophase` ∈ PHENOPHASES.
- [ ] No NaN.
