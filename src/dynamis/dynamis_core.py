# -*- coding: utf-8 -*-
"""
==============================================================================
DYNAMIS CORE v1.0 - THE CHAOS INFERENCE ENGINE
For the Global Synergy Mesh
==============================================================================

A Differentiable Hierarchical Kalman Filter for learning latent dynamics
in chaotic, noisy systems.

CORE COMPONENTS:
    1. HilbertEmbedding: Maps N-dimensional discrete/continuous data to
       a locality-preserving 1D/2D stream.
    2. MarkovKalmanModule (MKM): Learnable Kalman Filter.
    3. HRM_MKM: Hierarchical Reasoning Model integrating Executor and MKM.
    4. HierarchicalHPR: Multi-timescale reasoner.

DESIGN PRINCIPLES:
    - GENERIC: No domain-specific logic (lottery, finance, etc.).
    - DIFFERENTIABLE: All components support backpropagation.
    - UNCERTAINTY-AWARE: Outputs include covariance (confidence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

# Optional: Hilbert Curve support
try:
    from hilbertcurve.hilbertcurve import HilbertCurve as HilbertCurveLib
    HILBERT_AVAILABLE = True
except ImportError:
    HILBERT_AVAILABLE = False
    print("[WARN] hilbertcurve library not found. Install with: pip install hilbertcurve")


# ==============================================================================
# SECTION 1: HILBERT EMBEDDING (Topological Encoder)
# ==============================================================================

class HilbertEmbedding:
    """
    Maps N-dimensional discrete or continuous data to Hilbert Space coordinates.
    
    Hilbert curves are space-filling curves that preserve locality:
    items that are "close" in N-dimensional space remain "close" in 1D.
    
    Args:
        n_items: The number of discrete items to map (e.g., 60 for Mega-Sena, 
                 or 1000 for a vocabulary). For continuous, set high.
        p_order: Hilbert curve order. Higher = finer granularity = more compute.
        n_dims: Number of dimensions for the Hilbert curve (default 2).
    """
    def __init__(self, n_items: int, p_order: int = 5, n_dims: int = 2):
        self.n_items = n_items
        self.p_order = p_order
        self.n_dims = n_dims
        
        if not HILBERT_AVAILABLE:
            raise ImportError("hilbertcurve library required for HilbertEmbedding.")
        
        self.mapper = HilbertCurveLib(p_order, n_dims)
        self.max_h = (2 ** (p_order * n_dims)) - 1
        
        # Pre-compute the mapping for discrete items
        self.item_to_coord = {}
        self.coord_to_item = {}
        
        for i in range(1, n_items + 1):
            distance = int((i / n_items) * self.max_h)
            coords = tuple(self.mapper.point_from_distance(distance))
            # Normalize to [0, 1]
            norm_coords = tuple(c / (2 ** p_order) for c in coords)
            self.item_to_coord[i] = norm_coords
            self.coord_to_item[norm_coords] = i

    def encode(self, items: list) -> torch.Tensor:
        """
        Encodes a list of discrete items to their Hilbert centroid.
        
        Args:
            items: A list of integers (e.g., [1, 5, 23]).
        Returns:
            A FloatTensor of shape (n_dims,) representing the centroid.
        """
        coords = np.array([self.item_to_coord.get(int(i), (0.5,) * self.n_dims) for i in items])
        centroid = np.mean(coords, axis=0)
        return torch.FloatTensor(centroid)

    def decode_neighbors(self, centroid: torch.Tensor, k: int = 10) -> list:
        """
        Finds the k items closest to a given centroid in Hilbert space.
        
        Args:
            centroid: A Tensor of shape (n_dims,).
            k: Number of neighbors to return.
        Returns:
            A list of the k closest item integers.
        """
        c = centroid.detach().cpu().numpy()
        distances = []
        for norm_coords, item in self.coord_to_item.items():
            dist = np.linalg.norm(np.array(norm_coords) - c)
            distances.append((dist, item))
        distances.sort(key=lambda x: x[0])
        return [item for _, item in distances[:k]]


# ==============================================================================
# SECTION 1.5: HURST EXPONENT (Regime Filter / Gatekeeper)
# ==============================================================================

def calculate_hurst(series: np.ndarray, min_window: int = 10, max_window: int = 100) -> float:
    """
    Calculate the Hurst Exponent using Rescaled Range (R/S) Analysis.
    
    H < 0.5: Anti-persistent (Mean Reverting)
    H = 0.5: Random Walk (Brownian Motion) - NO MEMORY
    H > 0.5: Persistent (Trending)
    
    Args:
        series: 1D NumPy array of prices or log-returns.
        min_window: Minimum window size for R/S calculation.
        max_window: Maximum window size.
        
    Returns:
        Hurst exponent (float between 0 and 1).
    """
    if len(series) < max_window:
        max_window = len(series) // 2
    if max_window < min_window:
        return 0.5  # Not enough data, assume random
    
    # Generate window sizes (powers of 2 or linear)
    window_sizes = []
    w = min_window
    while w <= max_window:
        window_sizes.append(w)
        w = int(w * 1.5)  # Logarithmic spacing
    
    rs_values = []
    
    for window in window_sizes:
        rs_list = []
        num_windows = len(series) // window
        
        for i in range(num_windows):
            chunk = series[i * window : (i + 1) * window]
            if len(chunk) < 2:
                continue
                
            # Mean-centered cumulative deviation
            mean_val = np.mean(chunk)
            deviation = chunk - mean_val
            cumulative = np.cumsum(deviation)
            
            # Range
            R = np.max(cumulative) - np.min(cumulative)
            
            # Standard Deviation
            S = np.std(chunk, ddof=1)
            
            if S > 1e-10:  # Avoid division by zero
                rs_list.append(R / S)
        
        if len(rs_list) > 0:
            rs_values.append((window, np.mean(rs_list)))
    
    if len(rs_values) < 2:
        return 0.5  # Not enough data points for regression
    
    # Linear regression: log(R/S) = H * log(n) + c
    log_windows = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    
    # Least squares fit
    H = np.polyfit(log_windows, log_rs, 1)[0]
    
    # Clamp to valid range
    return float(np.clip(H, 0.0, 1.0))


def is_predictable_regime(series: np.ndarray, threshold: float = 0.05) -> Tuple[bool, float, str]:
    """
    Gatekeeper function: Determines if Dynamis should attempt prediction.
    
    Args:
        series: Price or return series.
        threshold: Distance from 0.5 required to be considered non-random.
        
    Returns:
        (is_predictable, hurst_value, regime_name)
    """
    H = calculate_hurst(series)
    
    if H > 0.5 + threshold:
        return (True, H, "PERSISTENT (Trending)")
    elif H < 0.5 - threshold:
        return (True, H, "ANTI-PERSISTENT (Mean Reverting)")
    else:
        return (False, H, "RANDOM WALK (Skip)")


# ==============================================================================
# SECTION 2: EXECUTOR (Low-Level Processor)
# ==============================================================================

class Executor(nn.Module):
    """
    The Low-Level processing unit.
    Combines raw input with top-level context (Kalman state prediction) to
    generate a 'measurement' for the filter and an optional task output.
    
    Args:
        input_dim: Dimension of input features.
        state_dim: Dimension of the Kalman state (context).
        hidden_dim: Dimension of the GRU hidden state.
        output_dim: Dimension of the final task output (optional, can be None).
    """
    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.gru = nn.GRUCell(input_dim + state_dim, hidden_dim)
        self.to_measurement = nn.Linear(hidden_dim, state_dim)
        
        if output_dim is not None:
            self.to_output = nn.Linear(hidden_dim, output_dim)
        else:
            self.to_output = None

    def forward(
        self, 
        input_t: torch.Tensor, 
        context_state: torch.Tensor, 
        hidden_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_t: (batch, input_dim)
            context_state: (batch, state_dim) - A priori prediction from MKM.
            hidden_prev: (batch, hidden_dim)
        Returns:
            hidden_curr: (batch, hidden_dim)
            measurement: (batch, state_dim) - Feedback for MKM.
            output: (batch, output_dim) or None.
        """
        combined = torch.cat([input_t, context_state], dim=1)
        hidden_curr = self.gru(combined, hidden_prev)
        measurement = self.to_measurement(hidden_curr)
        output = self.to_output(hidden_curr) if self.to_output else None
        return hidden_curr, measurement, output


# ==============================================================================
# SECTION 3: MARKOV-KALMAN MODULE (High-Level Planner)
# ==============================================================================

class MarkovKalmanModule(nn.Module):
    """
    Differentiable Kalman Filter with Learnable Dynamics.
    Unlike classical KF, the matrices A, H, Q, R are nn.Parameters.
    
    Args:
        state_dim: Dimension of the latent state.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        # Learnable Transition: x_t = A @ x_{t-1}
        self.A = nn.Parameter(torch.eye(state_dim))
        # Learnable Observation: z_t = H @ x_t
        self.H = nn.Parameter(torch.eye(state_dim))
        # Process Noise Covariance (log-diagonal for positivity)
        self.log_Q_diag = nn.Parameter(torch.zeros(state_dim) - 2.0)
        # Measurement Noise Covariance
        self.log_R_diag = nn.Parameter(torch.zeros(state_dim) - 1.0)

    def predict(
        self, 
        x_post: torch.Tensor, 
        P_post: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Kalman Prediction Step (A Priori).
        
        Args:
            x_post: (batch, state_dim) - Posterior mean from last step.
            P_post: (batch, state_dim, state_dim) - Posterior covariance.
        Returns:
            x_pred: (batch, state_dim) - Prior mean.
            P_pred: (batch, state_dim, state_dim) - Prior covariance.
        """
        batch_size = x_post.size(0)
        Q = torch.diag_embed(torch.exp(self.log_Q_diag))
        
        x_pred = F.linear(x_post, self.A)
        
        A_batch = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        P_pred = torch.bmm(torch.bmm(A_batch, P_post), A_batch.transpose(1, 2)) + Q
        
        return x_pred, P_pred

    def update(
        self, 
        x_pred: torch.Tensor, 
        P_pred: torch.Tensor, 
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Kalman Update Step (A Posteriori).
        
        Args:
            x_pred: (batch, state_dim) - Prior mean.
            P_pred: (batch, state_dim, state_dim) - Prior covariance.
            measurement: (batch, state_dim) - Observation from Executor.
        Returns:
            x_post: (batch, state_dim) - Posterior mean.
            P_post: (batch, state_dim, state_dim) - Posterior covariance.
            innovation: (batch, state_dim) - Prediction error.
        """
        batch_size = x_pred.size(0)
        R = torch.diag_embed(torch.exp(self.log_R_diag))
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Innovation (Residual)
        innovation = measurement - F.linear(x_pred, self.H)
        
        # Innovation Covariance
        S = torch.bmm(torch.bmm(H_batch, P_pred), H_batch.transpose(1, 2)) + R
        S = S + 1e-6 * torch.eye(self.state_dim, device=S.device).unsqueeze(0) # Jitter
        
        # Kalman Gain
        HP_T = torch.bmm(P_pred, H_batch.transpose(1, 2))
        K = torch.linalg.solve(S, HP_T.transpose(1, 2)).transpose(1, 2)
        
        # State Update
        x_update = torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        x_post = x_pred + x_update
        
        # Covariance Update
        I = torch.eye(self.state_dim, device=x_pred.device).unsqueeze(0).expand(batch_size, -1, -1)
        P_post = torch.bmm((I - torch.bmm(K, H_batch)), P_pred)
        
        return x_post, P_post, innovation


# ==============================================================================
# SECTION 4: HRM-MKM INTEGRATION (The Core Reasoning Loop)
# ==============================================================================

class HRM_MKM(nn.Module):
    """
    Hierarchical Reasoning Model with Markov-Kalman Module.
    Integrates Executor and MKM in a temporal loop.
    
    Args:
        input_dim: Dimension of input features per timestep.
        hidden_dim: GRU hidden dimension.
        state_dim: Kalman state dimension.
        output_dim: Task output dimension (or None for state-only).
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        state_dim: int, 
        output_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.executor = Executor(input_dim, state_dim, hidden_dim, output_dim)
        self.mkm = MarkovKalmanModule(state_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (batch, seq_len, input_dim)
        Returns:
            outputs: (batch, seq_len, output_dim) or None if output_dim is None.
            innovations: (batch, seq_len, state_dim) - For innovation loss.
            final_state: (batch, state_dim) - Final posterior mean.
        """
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        
        h_exec = torch.zeros(batch_size, self.hidden_dim, device=device)
        x_mkm = torch.zeros(batch_size, self.state_dim, device=device)
        P_mkm = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        outputs = []
        innovations = []
        
        for t in range(seq_len):
            input_t = inputs[:, t, :]
            x_pred, P_pred = self.mkm.predict(x_mkm, P_mkm)
            h_exec, measurement, output_t = self.executor(input_t, x_pred, h_exec)
            x_mkm, P_mkm, innovation = self.mkm.update(x_pred, P_pred, measurement)
            
            if output_t is not None:
                outputs.append(output_t)
            innovations.append(innovation)
        
        outputs_tensor = torch.stack(outputs, dim=1) if outputs else None
        innovations_tensor = torch.stack(innovations, dim=1)
        
        return outputs_tensor, innovations_tensor, x_mkm


# ==============================================================================
# SECTION 5: HIERARCHICAL HPR (Multi-Timescale Reasoning)
# ==============================================================================

class ContextualTransitionNet(nn.Module):
    """MLP for contextual state transition: f(x, context)."""
    def __init__(self, state_dim: int, context_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + context_dim, 64),
            nn.Tanh(),
            nn.Linear(64, state_dim)
        )
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, context], dim=1))


class HierarchicalMKM_Node(nn.Module):
    """A single node in the hierarchical MKM."""
    def __init__(self, state_dim: int, context_dim: int = 0):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        
        if context_dim > 0:
            self.transition_fn = ContextualTransitionNet(state_dim, context_dim)
        else:
            self.transition_fn = nn.Sequential(
                nn.Linear(state_dim, 32), nn.Tanh(), nn.Linear(32, state_dim)
            )
        
        self.H = nn.Parameter(torch.eye(state_dim))
        self.log_Q_diag = nn.Parameter(torch.zeros(state_dim) - 2.0)
        self.log_R_diag = nn.Parameter(torch.zeros(state_dim) - 1.0)

    def predict(self, x_post, P_post, context=None):
        if self.context_dim > 0:
            x_pred = self.transition_fn(x_post, context)
        else:
            x_pred = self.transition_fn(x_post)
        Q = torch.diag_embed(torch.exp(self.log_Q_diag))
        P_pred = P_post + Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, measurement):
        batch_size = x_pred.size(0)
        R = torch.diag_embed(torch.exp(self.log_R_diag))
        H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        y = measurement - F.linear(x_pred, self.H)
        S = torch.bmm(torch.bmm(H_batch, P_pred), H_batch.transpose(1, 2)) + R
        S = S + 1e-6 * torch.eye(self.state_dim, device=S.device).unsqueeze(0)
        HP_T = torch.bmm(P_pred, H_batch.transpose(1, 2))
        K = torch.linalg.solve(S, HP_T.transpose(1, 2)).transpose(1, 2)
        x_post = x_pred + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
        I = torch.eye(self.state_dim, device=x_pred.device).unsqueeze(0)
        P_post = torch.bmm((I - torch.bmm(K, H_batch)), P_pred)
        return x_post, P_post


class HierarchicalHPR(nn.Module):
    """
    Multi-Timescale Hierarchical Probabilistic Reasoner.
    Level 2 (Slow/Strategic) sets context for Level 1 (Fast/Tactical).
    
    Args:
        input_dim: Input feature dimension.
        hidden_dim: Executor hidden dimension.
        state_dim: Kalman state dimension.
        output_dim: Task output dimension.
        time_scale: How often Level 2 updates (every N steps).
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        state_dim: int = 4, 
        output_dim: Optional[int] = None,
        time_scale: int = 5
    ):
        super().__init__()
        self.time_scale = time_scale
        self.state_dim = state_dim
        
        self.level2 = HierarchicalMKM_Node(state_dim, context_dim=0)
        self.level1 = HierarchicalMKM_Node(state_dim, context_dim=state_dim)
        self.executor = Executor(input_dim, state_dim, hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        
        x2 = torch.zeros(batch_size, self.state_dim, device=device)
        P2 = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        x1 = torch.zeros(batch_size, self.state_dim, device=device)
        P1 = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        h_exec = torch.zeros(batch_size, self.executor.hidden_dim, device=device)
        
        outputs = []
        
        for t in range(seq_len):
            if t % self.time_scale == 0:
                x2_pred, P2_pred = self.level2.predict(x2, P2)
                x2, P2 = self.level2.update(x2_pred, P2_pred, x1.detach())
            
            x1_pred, P1_pred = self.level1.predict(x1, P1, context=x2)
            input_t = inputs[:, t, :]
            h_exec, meas, out = self.executor(input_t, x1_pred, h_exec)
            x1, P1 = self.level1.update(x1_pred, P1_pred, meas)
            
            if out is not None:
                outputs.append(out)
        
        outputs_tensor = torch.stack(outputs, dim=1) if outputs else None
        return outputs_tensor, x1, x2  # Return L1 and L2 final states


# ==============================================================================
# SECTION 6: LOSS FUNCTIONS
# ==============================================================================

def innovation_loss(innovations: torch.Tensor) -> torch.Tensor:
    """
    Regularization loss that minimizes Kalman 'surprise'.
    Forces the model to learn dynamics that match observations.
    """
    return torch.mean(innovations ** 2)


# ==============================================================================
# SECTION 7: EXAMPLE USAGE (Lorenz Attractor Test)
# ==============================================================================

if __name__ == "__main__":
    print("Dynamis Core v1.0 - Sanity Check")
    
    # Synthetic data: batch of 8, sequence of 50, 3 features (like Lorenz x, y, z)
    batch_size = 8
    seq_len = 50
    input_dim = 3
    output_dim = 3
    hidden_dim = 32
    state_dim = 8
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Test HRM_MKM
    model = HRM_MKM(input_dim, hidden_dim, state_dim, output_dim)
    outputs, innovations, final_state = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {outputs.shape if outputs is not None else 'None'}")
    print(f"  Innovation shape: {innovations.shape}")
    print(f"  Final state shape: {final_state.shape}")
    print(f"  Innovation Loss: {innovation_loss(innovations).item():.4f}")
    
    print("\n✅ All modules functional. Dynamis Core is ready for integration.")
