"""Data pipeline for Sentinel-2 multi-temporal crop classification."""
from .folder_consolidator import consolidate_regions, dates_in_order, summarise_consolidation
from .phenology_features import (
    N_PHENO_FEATURES,
    PHENO_FEATURE_NAMES,
    PhenologyFeatures,
    batch_phenology_features,
    compute_phenology_features,
)
from .point_extractor import extract_bands_at_point, extract_pixel_value
from .sample_strategy import (
    assign_region_to_points,
    index_region_bboxes,
    sample_summary,
    stratified_region_sample,
)
from .sentinel2_loader import (
    ALL_BANDS,
    MODEL_BANDS,
    TiffMeta,
    group_by_region_date,
    inventory_tiffs,
    parse_tiff_filename,
)
from .temporal_builder import (
    FEATURE_NAMES,
    N_FEATURES,
    PointSeries,
    build_point_series,
    build_training_set,
)
from .vegetation_indices import INDEX_NAMES, compute_all_indices, scale_l2a

__all__ = [
    "ALL_BANDS",
    "MODEL_BANDS",
    "TiffMeta",
    "parse_tiff_filename",
    "inventory_tiffs",
    "group_by_region_date",
    "consolidate_regions",
    "summarise_consolidation",
    "dates_in_order",
    "extract_pixel_value",
    "extract_bands_at_point",
    "INDEX_NAMES",
    "compute_all_indices",
    "scale_l2a",
    "FEATURE_NAMES",
    "N_FEATURES",
    "PointSeries",
    "build_point_series",
    "build_training_set",
    # sample strategy
    "index_region_bboxes",
    "assign_region_to_points",
    "stratified_region_sample",
    "sample_summary",
    # phenology features
    "PhenologyFeatures",
    "PHENO_FEATURE_NAMES",
    "N_PHENO_FEATURES",
    "compute_phenology_features",
    "batch_phenology_features",
]
