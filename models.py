"""
models.py — Black-box NN, PINN, and the physics-target integrator.
"""
import numpy as np
import torch
import torch.nn as nn
from config import (HIDDEN_DIM, N_HIDDEN_LAYERS, DT, RHS_T_CLAMP, LEARN_K0,
                    RK4_ONLINE_SUBSTEPS, PHYSICS_TARGET_SUBSTEPS,
                    physics_prior_params)


def _mlp(input_dim, output_dim, hidden_dim, n_hidden):
    layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class _StateTransitionNet(nn.Module):
    """Shared base: maps normalized (CA_k, T_k, Tc_k) -> normalized (CA_k+1, T_k+1)."""
    def __init__(self, hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN_LAYERS):
        super().__init__()
        self.net = _mlp(3, 2, hidden_dim, n_hidden)

    def forward(self, x):
        return self.net(x)

    def predict_denormalized(self, CA, T, Tc, norm_stats):
        x = np.array([[CA, T, Tc]], dtype=np.float32)
        x_norm = (x - norm_stats['X_mean']) / norm_stats['X_std']
        with torch.no_grad():
            y_norm = self.forward(torch.FloatTensor(x_norm)).numpy()
        y = y_norm * norm_stats['Y_std'] + norm_stats['Y_mean']
        return float(y[0, 0]), float(y[0, 1])


class BlackBoxNN(_StateTransitionNet):
    """Purely data-driven state-transition model."""


def cstr_rhs_torch(CA, T, Tc, params):
    """Vectorized CSTR right-hand side. Clamps are numerical safety only."""
    T_s = torch.clamp(T, *RHS_T_CLAMP)
    CA_s = torch.clamp(CA, min=0.0)
    rA = params['k0'] * torch.exp(-params['Ea_over_R'] / T_s) * CA_s
    q_V = params['q'] / params['V']
    dCA = q_V * (params['CA_f'] - CA_s) - rA
    dT = q_V * (params['T_f'] - T_s) + (-params['delta_H'] / params['rho_Cp']) * rA \
        + (params['UA'] / (params['rho_Cp'] * params['V'])) * (Tc - T_s)
    return dCA, dT


def rk4_flow(CA, T, Tc, params, n_sub, dt=DT):
    """Explicit RK4 over one sampling interval with n_sub sub-steps."""
    h = dt / n_sub
    for _ in range(n_sub):
        k1 = cstr_rhs_torch(CA, T, Tc, params)
        k2 = cstr_rhs_torch(CA + 0.5 * h * k1[0], T + 0.5 * h * k1[1], Tc, params)
        k3 = cstr_rhs_torch(CA + 0.5 * h * k2[0], T + 0.5 * h * k2[1], Tc, params)
        k4 = cstr_rhs_torch(CA + h * k3[0], T + h * k3[1], Tc, params)
        CA = CA + (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        T = T + (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return CA, T


def precompute_physics_targets(X_phys, params=None, n_sub=PHYSICS_TARGET_SUBSTEPS):
    """
    Phi_dt(x_k, u_k) for every row of X_phys = [CA, T, Tc] (physical units).
    Computed once, without gradients, in float64. Returns (N, 2) array.
    """
    params = physics_prior_params() if params is None else params
    X = torch.as_tensor(np.asarray(X_phys), dtype=torch.float64)
    with torch.no_grad():
        CA, T = rk4_flow(X[:, 0], X[:, 1], X[:, 2], params, n_sub)
    return torch.stack([CA, T], dim=1).numpy()


class PhysicsInformedNN(_StateTransitionNet):
    """
    Same network as BlackBoxNN, trained with an added one-step physics residual
    || f_theta(x_k, u_k) - Phi_dt(x_k, u_k) ||^2 (normalized by output std).
    """
    def __init__(self, norm_stats=None, hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN_LAYERS):
        super().__init__(hidden_dim, n_hidden)
        self.norm_stats = norm_stats
        self.params = physics_prior_params()
        self.log_k0 = nn.Parameter(torch.log(torch.tensor(float(self.params['k0']))),
                                   requires_grad=LEARN_K0)

    def _residual(self, x_norm, target_phys):
        Y_mean = torch.as_tensor(self.norm_stats['Y_mean'], dtype=torch.float32)
        Y_std = torch.as_tensor(self.norm_stats['Y_std'], dtype=torch.float32)
        y_pred = self.forward(x_norm) * Y_std + Y_mean
        err = (y_pred - target_phys) / Y_std
        return torch.mean(err[:, 0] ** 2 + err[:, 1] ** 2)

    def physics_loss_precomputed(self, x_norm, target_phys):
        """target_phys: (B, 2) float tensor of precomputed Phi values."""
        return self._residual(x_norm, target_phys)

    def physics_loss_online(self, x_norm):
        """Legacy mode: RK4 inside the graph (gradient flows to log_k0 if learned)."""
        X_mean = torch.as_tensor(self.norm_stats['X_mean'], dtype=torch.float32)
        X_std = torch.as_tensor(self.norm_stats['X_std'], dtype=torch.float32)
        x = x_norm * X_std + X_mean
        p = dict(self.params)
        p['k0'] = torch.exp(self.log_k0)
        CA, T = rk4_flow(x[:, 0], x[:, 1], x[:, 2], p, RK4_ONLINE_SUBSTEPS)
        return self._residual(x_norm, torch.stack([CA, T], dim=1))
