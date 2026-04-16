"""
Innovation Loss + Calibration (ECE) for Dynamis models.

The innovation vector is the Kalman "surprise" — the residual between observation
and prior prediction. Minimising squared innovations forces the model to learn
true dynamics rather than memorise outputs. The ECE term adds calibration pressure.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def innovation_loss(innovations: torch.Tensor) -> torch.Tensor:
    """Mean squared innovation across batch, time, and state dims."""
    return torch.mean(innovations**2)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """
    Differentiable ECE approximation over probabilistic predictions.

    Args:
        probs: (B, C) softmax probabilities.
        labels: (B,) int64 ground-truth class indices.
        n_bins: number of confidence bins.

    Returns:
        Scalar tensor, the weighted absolute gap between accuracy and confidence.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).float()

    ece = probs.new_zeros(())
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:], strict=True):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            weight = in_bin.float().mean()
            ece = ece + weight * (acc_in_bin - conf_in_bin).abs()
    return ece


def dynamis_loss(
    crop_logits: torch.Tensor,
    crop_labels: torch.Tensor,
    pheno_logits: torch.Tensor,
    pheno_labels: torch.Tensor,
    innovations: torch.Tensor,
    lambda_innovation: float = 0.10,
    lambda_ece: float = 0.05,
    class_weights_crop: torch.Tensor | None = None,
    class_weights_pheno: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Combined loss for the dual-head Dynamis classifier.

    Returns dict with: total, ce_crop, ce_pheno, innov, ece_crop, ece_pheno.
    """
    ce_crop = F.cross_entropy(crop_logits, crop_labels, weight=class_weights_crop)
    ce_pheno = F.cross_entropy(pheno_logits, pheno_labels, weight=class_weights_pheno)
    innov = innovation_loss(innovations)

    crop_probs = F.softmax(crop_logits, dim=-1)
    pheno_probs = F.softmax(pheno_logits, dim=-1)
    ece_crop = expected_calibration_error(crop_probs, crop_labels)
    ece_pheno = expected_calibration_error(pheno_probs, pheno_labels)

    total = (
        ce_crop
        + ce_pheno
        + lambda_innovation * innov
        + lambda_ece * (ece_crop + ece_pheno)
    )
    return {
        "total": total,
        "ce_crop": ce_crop.detach(),
        "ce_pheno": ce_pheno.detach(),
        "innov": innov.detach(),
        "ece_crop": ece_crop.detach(),
        "ece_pheno": ece_pheno.detach(),
    }


__all__ = ["innovation_loss", "expected_calibration_error", "dynamis_loss"]
