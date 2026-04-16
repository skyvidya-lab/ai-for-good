"""Dynamis — physics-informed modules for crop classification."""
from .chaos_attention import ChaosAttention
from .dynamis_core import (
    Executor,
    HierarchicalHPR,
    HilbertEmbedding,
    HRM_MKM,
    MarkovKalmanModule,
    calculate_hurst,
    is_predictable_regime,
)
from .hurst_geo import hurst_features, hurst_spectral, hurst_temporal
from .innovation_loss import dynamis_loss, expected_calibration_error, innovation_loss
from .phenology_prior import (
    N_PHENOPHASES,
    PHENO_TO_IDX,
    PHENOPHASES,
    build_phenology_prior_tensor,
    build_phenology_transition_matrix,
    phenophase_index_to_name,
    phenophase_name_to_index,
)

__all__ = [
    # core
    "HilbertEmbedding",
    "MarkovKalmanModule",
    "Executor",
    "HRM_MKM",
    "HierarchicalHPR",
    "calculate_hurst",
    "is_predictable_regime",
    # geo
    "hurst_temporal",
    "hurst_spectral",
    "hurst_features",
    # physics
    "ChaosAttention",
    # loss
    "innovation_loss",
    "expected_calibration_error",
    "dynamis_loss",
    # phenology
    "PHENOPHASES",
    "PHENO_TO_IDX",
    "N_PHENOPHASES",
    "build_phenology_transition_matrix",
    "build_phenology_prior_tensor",
    "phenophase_name_to_index",
    "phenophase_index_to_name",
]
