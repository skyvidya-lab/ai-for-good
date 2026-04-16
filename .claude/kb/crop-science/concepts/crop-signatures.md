# Crop Signatures (Rice / Corn / Soybean)

NE China summary for the 3 classes in this challenge.

## Rice

- **Planting**: flooding of paddies in May → unique NDWI/LSWI spike BEFORE Greenup.
- **Canopy**: transplanted seedlings → rapid rise to Peak by late Jul.
- **Physiology**: paddy water persists during tillering → NDWI stays high longer than other crops.
- **Signal discriminant**: the NDWI pre-Greenup spike is the **strongest single discriminator** from corn/soybean.
- **Harvest**: mid-Sep to early Oct, drained paddies → fast NDVI collapse.

## Corn (Maize)

- **Planting**: early May (dry sowing). No flooding signature.
- **Canopy**: C4 grass, very tall (~2 m) and dense → highest peak NDVI (often 0.85-0.92) of the three.
- **Physiology**: aggressive water uptake; fastest Greenup slope.
- **Signal discriminant**: extreme Peak value + fast rise rate.
- **Harvest**: late Sep.

## Soybean

- **Planting**: mid-to-late May.
- **Canopy**: legume (C3), lower (~0.6-1 m) than corn → peak NDVI ~0.75-0.85.
- **Physiology**: slower, more gradual growth; earlier Peak in late Jul; more gradual Senescence.
- **Signal discriminant**: lower peak + longer Maturity plateau.
- **Harvest**: late Sep to early Oct.

## Diagnostic chart (temporal NDVI)

```
NDVI ▲
1.0  |                     ____corn_____
0.8  |                   _/              \__
0.6  |                 _/ soybean           \__
     |               _/________________       \___
0.4  |            __/                            \___
     |     _____/                                    \__
0.2  | __/          rice (NDWI spike)
     |/_________________________________________________▶ date
     May  Jun  Jul  Aug  Sep  Oct
```

## Feature importance (hypothesised)

| Feature | Rice | Corn | Soybean |
|---|---|---|---|
| NDWI during Greenup | **HIGH** (flooding) | low | low |
| Peak NDVI magnitude | medium | **HIGH** | medium |
| Greenup slope | medium | **steep** | moderate |
| Senescence shape | abrupt | moderate | gradual |
| Hurst exponent | persistent Greenup | persistent Greenup | more random |

## Why Dynamis fits

- **MKM transition prior** captures that crop type modulates phenophase transition *rates*.
- **Innovation magnitude** differs systematically: rice has a "surprise" at pre-Greenup (paddy flood); corn has a "surprise" at Greenup onset (fast uptake); soybean is smoother.
- **Hurst** tags the persistent-growth segments differently for each crop.
