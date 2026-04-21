"""
Out-of-Distribution (OOD) detection for Dynamis.

v4's OOD signal relied solely on `trace(P)` (Kalman uncertainty) and
achieved AUC ~0.61 on the full-scale run — below the 0.7 target needed
to confidently route test points to the `background` class.

This module introduces a **combined score** that fuses three
complementary uncertainty signals:

1. `trace(P)` — Kalman posterior covariance (epistemic dynamics uncertainty).
2. Softmax entropy — classifier indecision across (rice, corn, soybean).
3. `innov_max` — largest Kalman innovation magnitude over the sequence
   (detects points whose spectral trajectory surprised the filter).

A logistic regression fits (trace_P, entropy, innov_max) → error-indicator
on out-of-fold validation data. The resulting score is a calibrated
probability of misclassification, which doubles as a principled
`background` detector: high score → route to `background`.

Falls back to `trace(P)` alone when too few errors exist to fit the LR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy H(p) = -sum p_i log p_i, per-row.

    Args:
        probs: (N, C) softmax probabilities.

    Returns:
        (N,) entropy in nats. Uniform over C=3 → log(3) ≈ 1.099.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


@dataclass
class OODModel:
    """Fitted logistic regressor that combines uncertainty signals."""

    coef: np.ndarray       # (3,) weights for [trace_P, entropy, innov_max]
    intercept: float
    feature_names: tuple[str, ...]
    mean: np.ndarray       # feature means (for standardisation)
    std: np.ndarray        # feature stds

    def score(self, trace_P: np.ndarray, entropy: np.ndarray, innov_max: np.ndarray) -> np.ndarray:
        """Return OOD probability in [0, 1]."""
        X = np.stack([trace_P, entropy, innov_max], axis=-1).astype(np.float64)
        Xn = (X - self.mean) / np.clip(self.std, 1e-8, None)
        logits = Xn @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))


def fit_combined_ood(
    trace_P: np.ndarray,
    entropy: np.ndarray,
    innov_max: np.ndarray,
    errors: np.ndarray,
    min_errors: int = 5,
) -> OODModel | None:
    """Fit a logistic regression on the three uncertainty features.

    Returns None (caller should fall back to raw trace_P) if there are
    fewer than `min_errors` errors — LR is useless on extreme class imbalance.

    Uses sklearn when available, a 50-step numpy gradient descent otherwise.
    """
    errors = np.asarray(errors, dtype=np.float64)
    if errors.sum() < min_errors or (len(errors) - errors.sum()) < min_errors:
        return None

    X = np.stack([trace_P, entropy, innov_max], axis=-1).astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Xn = (X - mean) / np.clip(std, 1e-8, None)

    # Try sklearn first (class_weight='balanced' handles imbalance)
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
        )
        lr.fit(Xn, errors)
        return OODModel(
            coef=lr.coef_[0].astype(np.float64),
            intercept=float(lr.intercept_[0]),
            feature_names=("trace_P", "entropy", "innov_max"),
            mean=mean,
            std=std,
        )
    except ImportError:
        # Minimal numpy fallback — weighted logistic via gradient descent
        pass

    # Numpy fallback: balanced class weights
    pos_w = 0.5 / max(errors.sum(), 1)
    neg_w = 0.5 / max((errors == 0).sum(), 1)
    w_per_sample = np.where(errors == 1, pos_w, neg_w) * len(errors)

    coef = np.zeros(3)
    intercept = 0.0
    lr = 0.1
    for _ in range(500):
        logits = Xn @ coef + intercept
        preds = 1.0 / (1.0 + np.exp(-logits))
        residual = (preds - errors) * w_per_sample
        coef -= lr * (Xn.T @ residual) / len(errors)
        intercept -= lr * residual.mean()
    return OODModel(
        coef=coef,
        intercept=intercept,
        feature_names=("trace_P", "entropy", "innov_max"),
        mean=mean,
        std=std,
    )


def combined_ood_score(
    trace_P: np.ndarray,
    probs: np.ndarray,
    innovations: np.ndarray | None = None,
    innov_max: np.ndarray | None = None,
    errors: np.ndarray | None = None,
    fitted_model: OODModel | None = None,
) -> tuple[np.ndarray, OODModel | None]:
    """Compute a combined OOD score per sample.

    Args:
        trace_P: (N,) Kalman trace from the model.
        probs: (N, C) softmax probabilities for the crop head.
        innovations: (N, T, state_dim) per-step innovations. Provide EITHER this
            or `innov_max`.
        innov_max: (N,) max-magnitude innovation per sample. Computed from
            `innovations` if None.
        errors: (N,) binary error indicator used to fit the logistic regressor.
            If None and `fitted_model` is None, falls back to raw trace_P.
        fitted_model: reuse an already-fitted OODModel (for inference-time
            application after training-time fitting).

    Returns:
        (scores, model_used_or_None)
    """
    trace_P = np.asarray(trace_P, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    entropy = softmax_entropy(probs)

    if innov_max is None:
        if innovations is None:
            raise ValueError("Provide either innovations or innov_max")
        innov_max = np.asarray(innovations, dtype=np.float64).__pow__(2).sum(axis=-1).max(axis=-1)
    else:
        innov_max = np.asarray(innov_max, dtype=np.float64)

    model = fitted_model
    if model is None and errors is not None:
        model = fit_combined_ood(trace_P, entropy, innov_max, errors)

    if model is None:
        # Fallback to normalised trace_P
        tp = (trace_P - trace_P.min()) / (trace_P.max() - trace_P.min() + 1e-8)
        return tp, None

    return model.score(trace_P, entropy, innov_max), model


__all__ = [
    "OODModel",
    "softmax_entropy",
    "fit_combined_ood",
    "combined_ood_score",
]
