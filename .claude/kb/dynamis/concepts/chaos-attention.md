# ChaosAttention (Physics-Informed Attention)

Multi-head attention whose score matrix is **additively modulated** by a per-sample physics-state vector. The idea: when the system is chaotic (high innovation, low Hurst), attention should widen; when the system is deterministic, attention should sharpen.

Implementation: [src/dynamis/chaos_attention.py](../../../../src/dynamis/chaos_attention.py). Adapted from `dynamis-finance-ai/backend/core/pi_attention.py` (finance PIA V4.2).

## The mechanism

Standard attention:

```
Attention = Softmax( Q K^T / √d ) V
```

ChaosAttention adds a physics bias that depends on the sample (not the tokens):

```
Attention = Softmax( Q K^T / √d + b_physics ) V

where  b_physics[h]  is  a  per-head  scalar  bias  produced  from:
    physics_state = [chaos_score, hurst]      (B, 2)
    b_physics = MLP(physics_state)            (B, num_heads)
```

- **chaos_score**: normalised mean of `‖innovation[t]‖²` over T — high = the filter is surprised a lot.
- **hurst**: the cascade Hurst value for this point.

The MLP (16 hidden units, GELU) learns how each head should respond to each physics state. No physics "prior" is hard-coded — backprop decides how to translate physics into attention sharpening vs widening.

## Why a scalar bias (not a rescaling)

We considered multiplying `Q K^T` by a physics-driven temperature. Rejected for two reasons:

1. A multiplicative "temperature" that's < 1 collapses the softmax; if learnt wrong, it makes attention degenerate.
2. Additive bias per head is a safer corridor — the softmax's relative ordering is preserved, it just shifts.

In practice the learned `b_physics` is small in magnitude — it's a **fine modulation**, not a takeover.

## Relation to HRM parent

Wang et al. 2025's HRM ([arXiv:2506.21734](https://arxiv.org/abs/2506.21734)) uses standard recurrent modules without physics conditioning. ChaosAttention is a Dynamis-specific **extension** on top of HRM-style architectures — the physics state is derived from the KF (which HRM doesn't have) and modulates the attention layer that pools the state trajectory.

In the Dynamis graph:

```
     Executor (GRU) ────► MKM (Kalman)
          │                   │
          │                   ▼
          │           state_trajectory + innovations
          │                   │
          ▼                   ▼
      hidden_seq     ┌─► physics_state: [chaos, hurst]
          │          │            │
          └──► concat│            ▼
                     │        ChaosAttention
                     ▼            │
               pooled_attn ◄──────┘
                     │
                     ▼
              physics_vec (v3)
                     │
                     ▼
                 head_crop
```

## When does it help?

From the v2→v3 A/B: ChaosAttention alone did not flip the result. What flipped it was **physics-vector injection into the crop head**. ChaosAttention's contribution is more subtle — it lets the pooled representation already reflect the physics state before we concatenate the raw physics vector.

Evidence: removing ChaosAttention and using plain attention degrades F1 macro by ~1pp on our sample (untested at full scale).

## Implementation details

[src/dynamis/chaos_attention.py](../../../../src/dynamis/chaos_attention.py):

```python
class ChaosAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, n_physics=2, dropout=0.1):
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.physics_adapter = nn.Sequential(
            nn.Linear(n_physics, 16),
            nn.GELU(),
            nn.Linear(16, num_heads),        # one scalar per head
        )
```

In the forward pass:

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
phys_bias = self.physics_adapter(physics_state)              # (B, H)
scores = scores + phys_bias.view(B, H, 1, 1)
if mask is not None:
    scores = scores.masked_fill(~mask_view, float('-inf'))
attn = F.softmax(scores, dim=-1)
```

The mask handling is standard — important because our sequences have padding for regions with fewer observations.

## Diagnostics

A healthy ChaosAttention produces physics biases in `[-2, +2]` (softmax-scale). If you see biases > 10 in magnitude, the physics adapter MLP is collapsing — retrain with lower lr or lower λ on the main loss.

Visualisation idea (v5+): plot `b_physics` vs `chaos_score` across samples — should be monotone if the attention learnt a coherent policy.

## Anti-patterns

- **Letting ChaosAttention be the ONLY physics path to the classifier** — v2 did this, and the physics signal got washed out by mean-pooling. Always also expose the raw physics vector (v3 fix).
- **Using too many physics inputs (n_physics > 4)** — the adapter MLP overfits on 184 samples. Keep to 2-3 scalars.
- **Forgetting the mask** — NDVI-missing timesteps were never re-checked in v1; the mask argument is now non-negotiable.

## References

- PIA V4.2 (finance origin): `C:\Users\eluzq\workspace\dynamis-finance-ai\backend\core\pi_attention.py`
- Bahdanau et al. 2014 — original attention mechanism (for baseline math).
- Vaswani et al. 2017 — multi-head attention (transformer foundation).
- Our tests — [tests/test_dynamis_modules.py::test_chaos_attention_forward_masked](../../../../tests/test_dynamis_modules.py).
