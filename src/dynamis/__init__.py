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
from .hurst_geo import (
    hurst_bounded,
    hurst_dfa,
    hurst_diff_regional,
    hurst_features,
    hurst_regional,
    hurst_spectral,
    hurst_temporal,
)
from .ood import OODModel, combined_ood_score, fit_combined_ood, softmax_entropy
from .innovation_loss import dynamis_loss, expected_calibration_error, innovation_loss
from .phenology_decoder import (
    estimate_greenup_anchor,
    get_crop_cumulative_days,
    viterbi_pheno_decode,
    viterbi_pheno_decode_batch,
)
from .phenology_prior import (
    N_PHENOPHASES,
    PHENO_TO_IDX,
    PHENOPHASES,
    PHENOPHASE_CUMULATIVE_DAYS,
    PHENOPHASE_INTERVALS_MEAN,
    PHENOPHASE_INTERVALS_STD,
    build_phenology_interval_embedding,
    build_phenology_prior_tensor,
    build_phenology_transition_matrix,
    get_crop_interval_prior,
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
    "hurst_regional",
    "hurst_bounded",
    "hurst_dfa",
    "hurst_diff_regional",
    "hurst_features",
    # ood
    "OODModel",
    "softmax_entropy",
    "fit_combined_ood",
    "combined_ood_score",
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
    "PHENOPHASE_CUMULATIVE_DAYS",
    "PHENOPHASE_INTERVALS_MEAN",
    "PHENOPHASE_INTERVALS_STD",
    "build_phenology_transition_matrix",
    "build_phenology_prior_tensor",
    "build_phenology_interval_embedding",
    "get_crop_interval_prior",
    "phenophase_name_to_index",
    "phenophase_index_to_name",
    # phenology decoder (V12)
    "estimate_greenup_anchor",
    "get_crop_cumulative_days",
    "viterbi_pheno_decode",
    "viterbi_pheno_decode_batch",
]
