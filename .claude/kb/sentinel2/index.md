# Sentinel-2 KB

## Concepts

- [bands.md](concepts/bands.md) — 13 bands, resolutions, wavelengths
- [vegetation-indices.md](concepts/vegetation-indices.md) — NDVI, EVI, NDWI, SAVI, LSWI formulas

## Product Level

This challenge uses **L2A (Bottom-Of-Atmosphere reflectance)** — already atmospherically corrected. Raw DN values are reflectance × 10 000 (scale factor). Divide by 10 000 before computing indices.

## Reference

- [Sentinel-2 MSI User Guide](https://sentiwiki.copernicus.eu/web/s2-mission)
