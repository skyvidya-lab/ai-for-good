# Submission Format

## Output CSV

Same columns as `test_point.csv`, with `Pre_crop_type` and `Pre_phenophase` filled:

```
point_id,Longitude,Latitude,phenophase_date,Pre_crop_type,Pre_phenophase
1,124.703696,48.543523,2018/9/1,corn,Maturity
2,125.331726,48.768485,2018/6/21,soybean,Greenup
...
```

## Reference (labelled sample)

`background/.../test_input_sample/test_data_label_sample.csv`:

```
point_id,Longitude,Latitude,phenophase_date,Pre_crop_type,Pre_phenophase
6,124.969305,48.806703,2018/7/23,corn,Maturity
7,124.914303,48.807416,2018/7/23,soybean,Maturity
8,124.903949,48.802374,2018/7/23,soybean,Maturity
9,124.807718,48.81456,2018/7/23,soybean,Maturity
```

## Rules of thumb

- **Date format**: `YYYY/M/D` (no zero-padding) — match the input.
- **Crop labels**: lowercase `rice`, `corn`, `soybean`, `background`.
- **Phenophase labels**: exact canonical spelling (see `src/dynamis/phenology_prior.py::PHENOPHASES`).
- **Order preserved**: output rows must match input row order (point_id, date) 1:1.
- **No extra columns**.

## Validation gate

Submit the baseline early (Phase 4 of the action plan) so the platform confirms format acceptance before we invest in Dynamis submission.
