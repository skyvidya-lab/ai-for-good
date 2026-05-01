"""
Top-3 Leaderboard Strategy: Specialized Rice Phenology Model (L-TAE)
Author: AI Assistant (Antigravity)
Date: 2026-05-01

This script implements the "Divide and Conquer" strategy for the Zero Hunger Challenge.
It uses a pure Transformer (L-TAE) with absolute Day-of-Year (DOY) positional encoding
and Gaussian Soft-Labels to solve the irregular sampling and sparse labels problems.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ==========================================
# 1. GAUSSIAN SOFT-LABELS (The 3% Solution)
# ==========================================

def get_doy(date_str):
    """Converts a date string to Day of Year (DOY)."""
    try:
        return pd.to_datetime(str(date_str)).dayofyear
    except:
        return 1  # fallback

def gaussian_soft_labels(ps_dates, phenophase_by_date, n_pheno, name_to_index_func, sigma=10.0):
    """
    Creates soft probability labels around exact observation dates using a Gaussian kernel.
    Returns:
        labels: (T, n_pheno) probability distribution.
        valid_mask: (T,) boolean mask where there is enough signal to train.
    """
    T = len(ps_dates)
    labels = np.zeros((T, n_pheno), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    if not phenophase_by_date:
        return labels, valid_mask

    ps_doy = np.array([get_doy(d) for d in ps_dates])
    
    events = []
    for k, v in phenophase_by_date.items():
        doy = get_doy(k)
        idx = name_to_index_func(v)
        events.append((doy, idx))

    for t, d_img in enumerate(ps_doy):
        max_w = 0.0
        for d_event, idx in events:
            # Minimal circular distance for year wrap-around
            dist = min(abs(d_img - d_event), 365 - abs(d_img - d_event))
            w = np.exp(-0.5 * (dist / sigma)**2)
            labels[t, idx] = max(labels[t, idx], w)
            max_w = max(max_w, w)
            
        # If the strongest probability is above 10%, we consider it a valid timestep to train
        if max_w > 0.1:
            labels[t] /= labels[t].sum()  # Normalize to sum=1
            valid_mask[t] = True

    return labels, valid_mask

# ==========================================
# 2. ARCHITECTURE: L-TAE (DOY Transformer)
# ==========================================

class PositionalEncodingDOY(nn.Module):
    def __init__(self, d_model, max_len=367):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, doy):
        """
        x: (B, T, D)
        doy: (B, T) LongTensor containing Day Of Year [1, 366]
        """
        doy = torch.clamp(doy, 0, 366)
        pe_batch = self.pe[doy]  # Shape: (B, T, d_model)
        return x + pe_batch

class SpecializedRicePhenoTransformer(nn.Module):
    def __init__(self, input_dim, n_pheno, d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.doy_encoding = PositionalEncodingDOY(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Predict ONLY Phenophase. Crop type is handled by LGBM.
        self.pheno_head = nn.Linear(d_model, n_pheno)

    def forward(self, x, doy, pad_mask=None):
        """
        x: (B, T, input_dim) - Raw satellite features
        doy: (B, T) - Day of Year integers
        pad_mask: (B, T) boolean tensor where True means "ignore this padding position"
        """
        h = self.input_proj(x)
        h = self.doy_encoding(h, doy)
        
        # Transformer encoder expects True for padded positions to ignore
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        
        logits = self.pheno_head(h)
        return logits

# ==========================================
# 3. SOFT CROSS ENTROPY LOSS
# ==========================================

def soft_cross_entropy(logits, soft_targets, mask):
    """
    Computes cross entropy using soft probability targets.
    logits: (B, T, n_pheno)
    soft_targets: (B, T, n_pheno) probability distributions
    mask: (B, T) boolean indicating valid timesteps
    """
    logits = logits[mask]
    soft_targets = soft_targets[mask]
    
    if len(logits) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    return loss.mean()

# ==========================================
# 4. UNIT TESTS AND MOCK EXECUTION
# ==========================================

def test_pipeline():
    print("--- Running Tests: Top-3 Rice Phenology Pipeline ---")
    
    # MOCK DATA
    B, T, input_dim = 2, 50, 21 # Batch=2, 50 timesteps, 21 features (Agro)
    n_pheno = 13
    
    # 1. Test Gaussian Soft Labels logic
    ps_dates = ['2018-01-10', '2018-06-05', '2018-06-15', '2018-12-30']
    pheno_dict = {'2018-06-10': 'Heading'} 
    def mock_name_to_idx(n): return 5 
    
    soft_labels, valid_mask = gaussian_soft_labels(ps_dates, pheno_dict, n_pheno, mock_name_to_idx, sigma=10.0)
    print("Dates DOY:", [get_doy(d) for d in ps_dates])
    print("Observation DOY:", get_doy('2018-06-10'))
    print("Valid Training Timesteps:", valid_mask)
    print(f"Soft Label (idx 5) at Jun 05: {soft_labels[1, 5]:.3f}")
    print(f"Soft Label (idx 5) at Jun 15: {soft_labels[2, 5]:.3f}")
    assert soft_labels[1, 5] > 0 and soft_labels[2, 5] > 0, "Gaussian Smoothing Failed"
    
    # 2. Test Transformer Forward Pass
    model = SpecializedRicePhenoTransformer(input_dim, n_pheno)
    
    x_mock = torch.randn(B, T, input_dim)
    doy_mock = torch.randint(1, 366, (B, T))
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[:, 40:] = True 
    
    logits = model(x_mock, doy_mock, pad_mask)
    print(f"\nTransformer output shape: {logits.shape} (Expected: {B}, {T}, {n_pheno})")
    assert logits.shape == (B, T, n_pheno), "Transformer shape mismatch"
    
    # 3. Test Soft Loss
    soft_targets = F.softmax(torch.randn(B, T, n_pheno), dim=-1)
    valid_train_mask = torch.rand(B, T) > 0.8
    
    loss = soft_cross_entropy(logits, soft_targets, valid_train_mask)
    print(f"Soft Cross Entropy Loss computed: {loss.item():.4f}")
    
    print("\n[SUCCESS] All core components passed tests. Architecture is ready for training.")

if __name__ == "__main__":
    try:
        import torch
        test_pipeline()
    except ImportError:
        print("[ERROR] PyTorch is not installed in the local environment.")
        print("Please upload/run this script in your GPU environment (Colab/Lightning).")
