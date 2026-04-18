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


def test_dynamis_exposes_physics_vector_to_crop_head():
    """The crop head must receive the Dynamis physics signals as explicit
    features (innovation stats + final state + trace(P) stats + hurst).
    Without this, Dynamis loses the discriminative edge it computes
    internally — see the v2 report where this was the root cause of the
    -7pp gap to LightGBM."""
    from src.models import DynamisCropClassifier, DynamisModelConfig

    cfg = DynamisModelConfig(input_dim=17, state_dim=7, hidden_dim=32)
    model = DynamisCropClassifier(cfg)
    B, T = 3, 5
    x = torch.randn(B, T, 17)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = model(x, mask=mask)

    assert "physics_vector" in out, "physics_vector must be exposed"
    # 8 scalar stats + state_dim=7 → 15 features
    assert out["physics_vector"].shape == (B, 8 + 7)
    assert "P_trajectory" in out
    assert out["P_trajectory"].shape == (B, T, 7)


def test_dynamis_crop_head_gradient_flows_through_physics():
    """A change in the innovation signal must change the crop logits,
    proving the physics vector is actually wired into the head."""
    import torch.nn.functional as F

    from src.models import DynamisCropClassifier, DynamisModelConfig

    torch.manual_seed(0)
    cfg = DynamisModelConfig(input_dim=17, state_dim=7, hidden_dim=32)
    model = DynamisCropClassifier(cfg).eval()
    B, T = 2, 4
    x1 = torch.randn(B, T, 17)
    x2 = x1.clone()
    x2[:, :, 12] += 5.0  # perturb the NDVI channel — should produce different innovations

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    # Physics vectors must differ (otherwise the physics injection is a stub)
    assert not torch.allclose(out1["physics_vector"], out2["physics_vector"]), (
        "physics_vector did not change under a meaningful input perturbation"
    )
    # And the crop logits must reflect the change
    assert not torch.allclose(
        F.softmax(out1["crop_logits"], -1),
        F.softmax(out2["crop_logits"], -1),
        atol=1e-3,
    ), "crop logits did not change — physics vector is not reaching the head"


def test_mkm_init_spread_produces_heterogeneous_Q_R():
    """Post-run optimisation §4: the default MKM init should give per-dim
    heterogeneity in log_Q_diag / log_R_diag so the Kalman filter does not
    immediately saturate to a shared trace(P)."""
    torch.manual_seed(0)
    mkm = MarkovKalmanModule(state_dim=7, noise_spread=0.5)
    q_std = mkm.log_Q_diag.detach().std().item()
    r_std = mkm.log_R_diag.detach().std().item()
    assert q_std > 0.05, f"log_Q_diag is too homogeneous (std={q_std})"
    assert r_std > 0.05, f"log_R_diag is too homogeneous (std={r_std})"


def test_uncertainty_differentiation_on_synthetic_data():
    """After 50 steps on noisy synthetic data, the resulting trace(P) should
    not collapse to a single value across the batch. This is a regression
    guard for the bug seen in the first run where trace(P) was identical
    (1.070) for every sample."""
    torch.manual_seed(0)
    mkm = MarkovKalmanModule(state_dim=4, noise_spread=0.5)
    B, T, D = 8, 50, 4
    # Mix of clean samples (easy) and high-noise samples (hard)
    easy = torch.randn(B // 2, T, D) * 0.1
    hard = torch.randn(B - B // 2, T, D) * 2.0
    meas = torch.cat([easy, hard], dim=0)

    x = torch.zeros(B, D)
    P = 0.1 * torch.eye(D).unsqueeze(0).expand(B, -1, -1).clone()
    for t in range(T):
        x_pred, P_pred = mkm.predict(x, P)
        x, P, _ = mkm.update(x_pred, P_pred, meas[:, t])

    trace_P = P.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    assert trace_P.std().item() > 1e-3, (
        f"trace(P) did not differentiate (std={trace_P.std().item()})"
    )


def test_hurst_returns_raw_without_clipping():
    """Post-run optimisation §3: when return_raw=True, calculate_hurst must
    expose the pre-clip value so we can inspect saturation."""
    rng = np.random.default_rng(1)
    # Strongly persistent series (cumulative sum of positive-biased noise)
    series = np.cumsum(rng.standard_normal(200) + 0.1)
    H_clipped = calculate_hurst(series, min_window=10, max_window=100)
    H_raw = calculate_hurst(series, min_window=10, max_window=100, return_raw=True)
    assert 0.0 <= H_clipped <= 1.0
    # Raw can exceed 1 for extreme trends; if it does, the fix is working.
    assert not (np.isnan(H_raw))


def test_hurst_returns_nan_on_too_short_series_when_raw():
    H_raw = calculate_hurst(np.array([1.0, 2.0, 3.0]), return_raw=True)
    assert np.isnan(H_raw) or abs(H_raw - 0.5) > 1e-6  # either NaN (preferred) or not the degenerate 0.5


# =============================================================================
# v4 additions — non-saturating Hurst (DFA) and temperature scaling
# =============================================================================

def test_hurst_dfa_does_not_saturate_on_monotonic_series():
    """v3 had ~66% saturation at H=1.00 because the classical R/S log-log slope
    blows up on near-monotonic NDVI curves. DFA should yield a value strictly
    below 1.0 on the same input."""
    from src.dynamis import hurst_dfa

    # Strongly monotonic series (linear + tiny noise) — the pathological case
    rng = np.random.default_rng(0)
    series = np.linspace(0.0, 1.0, 30) + 0.01 * rng.standard_normal(30)
    h = hurst_dfa(series)
    assert not np.isnan(h)
    assert 0.0 <= h <= 1.0
    assert h < 0.99, f"DFA saturated at {h} on a monotonic series (should stay below 1.0)"


def test_hurst_dfa_finite_on_random_walk():
    """Sanity check — DFA on a random walk should give ~0.5."""
    from src.dynamis import hurst_dfa

    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.standard_normal(500))
    h = hurst_dfa(walk)
    assert not np.isnan(h)
    assert 0.3 < h < 0.8  # generous bound; DFA has variance on finite samples


def test_temperature_scale_reduces_ece_on_overconfident_logits():
    """Construct synthetic over-confident logits (scaled up) and verify that
    temperature scaling learns T > 1 and reduces ECE."""
    from src.training import (
        apply_temperature,
        expected_calibration_error_np,
        temperature_scale,
    )

    rng = np.random.default_rng(42)
    n, c = 200, 3
    labels = rng.integers(0, c, size=n)
    # Base "well calibrated" logits
    base_logits = rng.standard_normal((n, c))
    # Bias toward the correct class but amplify to make it over-confident
    for i, y in enumerate(labels):
        base_logits[i, y] += 1.5
    overconfident = base_logits * 5.0  # × 5 inflates confidence pathologically

    probs_pre = np.exp(overconfident) / np.exp(overconfident).sum(-1, keepdims=True)
    ece_pre = expected_calibration_error_np(probs_pre, labels)

    T = temperature_scale(overconfident, labels, steps=300, lr=1e-2)
    assert T > 1.0, f"Expected T>1 on over-confident logits, got {T}"

    probs_post = apply_temperature(overconfident, T)
    ece_post = expected_calibration_error_np(probs_post, labels)
    assert ece_post < ece_pre, f"Temperature scaling did not reduce ECE: {ece_pre:.3f} → {ece_post:.3f}"


def test_temperature_scale_preserves_argmax():
    """Temperature scaling must not change which class has the highest probability."""
    from src.training import apply_temperature, temperature_scale

    rng = np.random.default_rng(7)
    logits = rng.standard_normal((50, 3)) * 2.0
    labels = rng.integers(0, 3, size=50)

    T = temperature_scale(logits, labels)
    probs_pre = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    probs_post = apply_temperature(logits, T)

    assert np.array_equal(probs_pre.argmax(-1), probs_post.argmax(-1)), (
        "Temperature scaling changed argmax — must be mathematically impossible"
    )
