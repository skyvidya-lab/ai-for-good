# Pattern: Linear Kalman Filter (reference implementation)

Canonical non-differentiable KF in ~40 lines. Use this when:

- Debugging a divergence bug (compare against a known-correct implementation).
- Teaching / explaining the filter to a stakeholder.
- Building a quick prototype before the learnable version.

## Code

```python
import numpy as np


class LinearKalmanFilter:
    """Minimal textbook Kalman filter for the linear-Gaussian case."""

    def __init__(self, A, H, Q, R, x0, P0):
        self.A, self.H, self.Q, self.R = A, H, Q, R
        self.x, self.P = x0, P0

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P

    def update(self, z):
        y = z - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x, self.P, y

    def step(self, z):
        self.predict()
        return self.update(z)
```

## Usage — scalar tracking (1-D Labbe ch. 04)

```python
kf = LinearKalmanFilter(
    A=np.array([[1.0]]),
    H=np.array([[1.0]]),
    Q=np.array([[0.01]]),
    R=np.array([[1.0]]),
    x0=np.zeros(1),
    P0=np.eye(1),
)

true = np.cumsum(np.random.randn(100) * 0.1)           # random walk
obs = true + np.random.randn(100)                      # noisy observations

estimates = []
for z in obs:
    x, P, y = kf.step(z.reshape(1))
    estimates.append(x[0])

# estimates should be smooth, between the noisy obs and the truth
```

## Usage — phenology-like 7-state

```python
# Tridiagonal A (self-loop + forward)
A = np.zeros((7, 7))
for i in range(7):
    A[i, i] = 0.7
    if i + 1 < 7:
        A[i, i+1] = 0.3

kf = LinearKalmanFilter(
    A=A,
    H=np.eye(7),                          # observe state directly
    Q=0.01 * np.eye(7),
    R=0.1 * np.eye(7),
    x0=np.array([1, 0, 0, 0, 0, 0, 0]),   # starts in Dormancy
    P0=0.1 * np.eye(7),
)

# Synthetic observations over 10 timesteps
for t in range(10):
    z = ...                                # from data
    kf.step(z)
```

## Numerical stability notes

Textbook `P = (I - KH) P` is unstable when `K` is near zero or `P` is near-singular. Use the **Joseph form**:

```python
I = np.eye(n)
IKH = I - K @ H
self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
```

Guaranteed symmetric and positive semi-definite. The textbook form is fine for learning but use Joseph in production.

## When NOT to use this

- Non-linear dynamics — use EKF or UKF.
- Non-Gaussian noise — use a particle filter.
- Need learnable parameters — use our `MarkovKalmanModule` from [src/dynamis/dynamis_core.py](../../../../src/dynamis/dynamis_core.py).
- Need batched inference (many filters in parallel) — write a vectorised version with `np.einsum` or use PyTorch `bmm`.

## Relation to our MKM

Our learnable version is essentially this code with:
1. `A`, `H`, `Q`, `R` replaced by `nn.Parameter`.
2. `np.linalg.inv` → `torch.linalg.solve` (more stable).
3. Batched: `(B, state_dim)` and `(B, state_dim, state_dim)` shapes throughout.
4. A GRU-based `Executor` that produces the measurement from raw features before the KF update.

## References

- Labbe ch. 04, 05, 06.
- `filterpy.kalman.KalmanFilter` — essentially the same implementation, battle-tested.
- Kalman 1960 — original paper, 11 pages, accessible.
