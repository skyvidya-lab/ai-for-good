"""
Post-hoc probability calibration.

Addresses the v3 pathology where Dynamis was over-confident (ECE=0.161):
a single learnable temperature scalar divides the logits before softmax.
Fit on held-out (out-of-fold) predictions; no retraining required.

Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks"
(https://arxiv.org/abs/1706.04599).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def temperature_scale(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    steps: int = 300,
    lr: float = 1e-2,
    device: str = "cpu",
) -> float:
    """Learn a single temperature `T` that minimises NLL on (logits, labels).

    Args:
        logits: (N, C) pre-softmax logits.
        labels: (N,) int class indices in [0, C).
        steps: LBFGS max iterations.
        lr: LBFGS step size.
        device: where to run (CPU is fine — it's a 1-scalar optimisation).

    Returns:
        Scalar `T`. Apply as `calibrated_probs = softmax(logits / T)`.
    """
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)

    T = torch.ones(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().item())


def apply_temperature(
    logits: np.ndarray | torch.Tensor,
    T: float,
) -> np.ndarray:
    """Return softmax probs after dividing logits by `T`."""
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    return F.softmax(logits_t / max(T, 1e-3), dim=-1).cpu().numpy()


def expected_calibration_error_np(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE from probabilities (numpy, reusable outside torch)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=-1)
    correct = (probs.argmax(-1) == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    weighted_gap = 0.0
    total = 0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = conf[in_bin].mean()
        weighted_gap += n * abs(acc_bin - conf_bin)
        total += n
    return float(weighted_gap / max(total, 1))


__all__ = [
    "temperature_scale",
    "apply_temperature",
    "expected_calibration_error_np",
]
