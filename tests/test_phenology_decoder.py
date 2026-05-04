"""Tests for the V12 crop-conditioned Viterbi phenology decoder."""
from __future__ import annotations

import numpy as np

from src.dynamis import (
    PHENOPHASES,
    estimate_greenup_anchor,
    get_crop_cumulative_days,
    viterbi_pheno_decode,
)


def _make_logits_for_sequence(doy: np.ndarray, anchor: float, crop: str, sharpness: float = 5.0):
    """Build synthetic logits where each timestep favors the canonically-expected phase."""
    cumulative = get_crop_cumulative_days(crop)
    T, K = len(doy), len(PHENOPHASES)
    logits = np.zeros((T, K), dtype=np.float32)
    for t, d in enumerate(doy):
        diffs = np.abs((d - anchor) - cumulative)
        diffs = np.minimum(diffs, 365.0 - diffs)
        logits[t] = -sharpness * diffs / 30.0
    return logits


def test_get_crop_cumulative_monotonic():
    cum = get_crop_cumulative_days('rice')
    assert cum.shape == (7,)
    assert cum[0] == 0.0
    assert np.all(np.diff(cum) > 0)


def test_estimate_anchor_returns_doy_of_max_greenup():
    doy = np.array([100, 130, 160, 190, 220], dtype=np.int64)
    valid = np.ones(5, dtype=bool)
    logits = np.zeros((5, 7), dtype=np.float32)
    logits[1, 0] = 5.0  # peak Greenup confidence at index 1 (doy=130)
    assert estimate_greenup_anchor(logits, doy, valid) == 130.0


def test_estimate_anchor_handles_no_valid():
    doy = np.array([100, 130], dtype=np.int64)
    valid = np.zeros(2, dtype=bool)
    logits = np.zeros((2, 7), dtype=np.float32)
    assert estimate_greenup_anchor(logits, doy, valid) is None


def test_viterbi_recovers_canonical_sequence_rice():
    anchor = 130.0
    cum = get_crop_cumulative_days('rice')
    doy = (anchor + cum).astype(np.int64)         # one obs per phase, on the expected day
    valid = np.ones(len(doy), dtype=bool)
    logits = _make_logits_for_sequence(doy, anchor, 'rice', sharpness=8.0)
    decoded = viterbi_pheno_decode(logits, doy, valid, crop_type='rice')
    assert decoded.tolist() == list(range(len(PHENOPHASES)))


def test_viterbi_invalid_steps_get_minus_100():
    doy = np.array([100, 0, 160], dtype=np.int64)
    valid = np.array([True, False, True], dtype=bool)
    logits = np.zeros((3, 7), dtype=np.float32)
    logits[0, 0] = 3.0
    logits[2, 1] = 3.0
    decoded = viterbi_pheno_decode(logits, doy, valid, crop_type='rice')
    assert decoded[1] == -100
    assert decoded[0] != -100
    assert decoded[2] != -100


def test_viterbi_crop_conditioning_changes_late_phases():
    """Soybean has senescence intervals ~8d shorter than rice — at a DOY around
    soybean's expected MidSenescence the decoder should disagree between crops."""
    rice_cum = get_crop_cumulative_days('rice')
    soy_cum = get_crop_cumulative_days('soybean')
    anchor = 130.0
    # ambiguous DOYs near the crop-specific late phases
    doy = (anchor + np.linspace(0, 130, 10)).astype(np.int64)
    valid = np.ones(len(doy), dtype=bool)
    # near-flat logits → timing prior dominates
    logits = np.zeros((len(doy), 7), dtype=np.float32)
    decoded_rice = viterbi_pheno_decode(logits, doy, valid, crop_type='rice')
    decoded_soy = viterbi_pheno_decode(logits, doy, valid, crop_type='soybean')
    # At least one timestep must differ given priors differ by ~8 days late season
    assert (decoded_rice != decoded_soy).any(), (
        f"Expected per-crop divergence. rice={decoded_rice} soy={decoded_soy} "
        f"rice_cum={rice_cum} soy_cum={soy_cum}"
    )
