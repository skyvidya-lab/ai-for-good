# Vegetation Indices

All formulas operate on reflectance values in `[0, 1]` (after `scale_l2a()`).

## NDVI — Normalized Difference Vegetation Index
```
NDVI = (NIR - RED) / (NIR + RED)    = (B08 - B04) / (B08 + B04)
```
Range: `[-1, 1]`. Higher = more vegetation. Core feature for all crops.

## EVI — Enhanced Vegetation Index
```
EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
```
Better than NDVI in dense canopies (saturates less). Sensitive to LAI.

## NDWI — Normalized Difference Water Index (McFeeters)
```
NDWI = (GREEN - NIR) / (GREEN + NIR)    = (B03 - B08) / (B03 + B08)
```
High values = open water. Useful for rice paddies during flooding.

## SAVI — Soil-Adjusted Vegetation Index
```
SAVI = (1 + L) * (NIR - RED) / (NIR + RED + L)    , L=0.5
```
Reduces soil brightness bias — important early in season (low canopy cover).

## LSWI — Land Surface Water Index
```
LSWI = (NIR - SWIR1) / (NIR + SWIR1)    = (B08 - B11) / (B08 + B11)
```
Canopy water content. Rice paddies show distinctive LSWI spikes during flooding that corn/soybean do not.

## Phenophase signatures

| Phase | NDVI | NDWI | LSWI | EVI |
|---|---|---|---|---|
| Dormancy | low (~0.1-0.3) | variable | low | low |
| Greenup | rising | low | rising | rising |
| MidGreenup | rising fast | low | rising | rising fast |
| Peak | max (~0.7-0.9) | lowest | max | max |
| Maturity | plateau | low | plateau | plateau |
| MidSenescence | declining | variable | declining | declining |
| Senescence | low | variable | low | low |

## Crop signatures (summary)

- **Rice**: NDWI/LSWI spike pre-Greenup (flooding); steep rise; distinct Peak plateau.
- **Corn**: very steep Greenup; high Peak (>0.8 NDVI); rapid Maturity→Senescence.
- **Soybean**: slower, lower NDVI curve; earlier Peak; smoother decline.

See [crop-science/concepts/crop-signatures.md](../../crop-science/concepts/crop-signatures.md) for details.
