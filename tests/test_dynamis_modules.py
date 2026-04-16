"""Smoke tests for Dynamis modules — structural sanity on synthetic tensors."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.dynamis import (
    ChaosAttention,
    MarkovKalmanModule,
    PHENOPHASES,
    N_PHENOPHASES,
    build_phenology_prior_tensor,
    build_phenology_transition_matrix,
    calculate_hurst,
    dynamis_loss,
    hurst_features,
    innovation_loss,
    phenophase_index_to_name,
    phenophase_name_to_index,
)


def test_phenology_prior_shape_and_row_stochastic():
    A = build_phenology_transition_matrix()
    assert A.shape == (N_PHENOPHASES, N_PHENOPHASES)
    assert np.allclose(A.sum(axis=1), 1.0)
    # Forward-only structure: no backward transitions except the Senescence→Dormancy wrap
    for i in range(N_PHENOPHASES - 1):
        for j in range(i):
            assert A[i, j] == 0.0, f"Backward transition at ({i},{j}): {A[i,j]}"


def test_phenology_prior_tensor_matches_numpy():
    A = build_phenology_transition_matrix()
    T = build_phenology_prior_tensor()
    assert torch.allclose(T, torch.as_tensor(A, dtype=torch.float32))


def test_phenophase_name_lookup_roundtrip():
    for i, name in enumerate(PHENOPHASES):
        assert phenophase_name_to_index(name) == i
        assert phenophase_index_to_name(i) == name
    # Case insensitive
    assert phenophase_name_to_index("dormancy") == 0


def test_hurst_on_random_walk_is_near_half():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal(500))
    h = calculate_hurst(walk, min_window=10, max_window=100)
    assert 0.3 < h < 0.7, f"Hurst on random walk should be ~0.5, got {h}"


def test_hurst_features_fallback():
    rng = np.random.default_rng(0)
    ndvi_short = np.array([0.3, 0.5, 0.7])  # too short for temporal Hurst
    bands = rng.random((3, 13))
    feats = hurst_features(ndvi_short, bands, min_temporal_dates=6)
    assert feats["hurst_temporal_valid"] == 0.0
    assert 0.0 <= feats["hurst_spectral_mean"] <= 1.0


def test_mkm_rollout_shapes():
    state_dim = 7
    mkm = MarkovKalmanModule(state_dim=state_dim)
    B = 4
    x_post = torch.zeros(B, state_dim)
    P_post = torch.eye(state_dim).unsqueeze(0).expand(B, -1, -1).clone()
    x_pred, P_pred = mkm.predict(x_post, P_post)
    assert x_pred.shape == (B, state_dim)
    assert P_pred.shape == (B, state_dim, state_dim)

    meas = torch.randn(B, state_dim)
    x_upd, P_upd, innov = mkm.update(x_pred, P_pred, meas)
    assert x_upd.shape == (B, state_dim)
    assert P_upd.shape == (B, state_dim, state_dim)
    assert innov.shape == (B, state_dim)


def test_chaos_attention_forward_masked():
    attn = ChaosAttention(embed_dim=32, num_heads=4, n_physics=2)
    B, T, D = 2, 5, 32
    x = torch.randn(B, T, D)
    physics = torch.rand(B, 2)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = attn(x, physics_state=physics, attn_mask=mask)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()


def test_innovation_loss_is_positive():
    innov = torch.randn(3, 5, 7)
    assert innovation_loss(innov).item() > 0


def test_dynamis_loss_total_includes_all_terms():
    B, T = 4, 3
    crop_logits = torch.randn(B, 3, requires_grad=True)
    pheno_logits = torch.randn(B * T, N_PHENOPHASES, requires_grad=True)
    crop_labels = torch.randint(0, 3, (B,))
    pheno_labels = torch.randint(0, N_PHENOPHASES, (B * T,))
    innov = torch.randn(B, T, N_PHENOPHASES, requires_grad=True)
    loss = dynamis_loss(crop_logits, crop_labels, pheno_logits, pheno_labels, innov)
    assert "total" in loss and loss["total"].requires_grad
    loss["total"].backward()
    assert crop_logits.grad is not None


def test_dynamis_crop_classifier_forward():
    from src.models import DynamisCropClassifier, DynamisModelConfig

    cfg = DynamisModelConfig(input_dim=17, state_dim=7, hidden_dim=32, attn_heads=4, n_crops=3)
    model = DynamisCropClassifier(cfg)
    B, T = 2, 4
    x = torch.randn(B, T, 17)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = model(x, mask=mask)
    assert out["crop_logits"].shape == (B, 3)
    assert out["pheno_logits"].shape == (B, T, 7)
    assert out["innovations"].shape == (B, T, 7)
    assert out["uncertainty"].shape == (B,)
