"""
cstr_plant.py — High-fidelity CSTR simulator (ground truth) + steady-state helper.
"""
import numpy as np
from scipy.integrate import solve_ivp
from config import CSTR_PARAMS, DT


def _p(params, key):
    """Plant parameter, honouring disturbance overrides."""
    alias = {'CA_f': 'CA_f_disturbed', 'T_f': 'T_f_disturbed', 'k0': 'k0_true'}
    if key in alias and alias[key] in params:
        return params[alias[key]]
    return params[key]


def cstr_ode(t, state, Tc, params=None):
    """[dCA/dt, dT/dt] for the jacketed exothermic CSTR."""
    params = CSTR_PARAMS if params is None else params
    CA, T = state
    q, V = params['q'], params['V']
    CA_f, T_f, k0 = _p(params, 'CA_f'), _p(params, 'T_f'), _p(params, 'k0')
    rA = k0 * np.exp(-params['Ea_over_R'] / T) * CA
    dCA = (q / V) * (CA_f - CA) - rA
    dT = (q / V) * (T_f - T) + (-params['delta_H'] / params['rho_Cp']) * rA \
        + (params['UA'] / (params['rho_Cp'] * V)) * (Tc - T)
    return [dCA, dT]


def simulate_cstr_step(state, Tc, dt=DT, params=None):
    """One sampling interval with zero-order-hold input (RK45, tight tolerances)."""
    sol = solve_ivp(cstr_ode, [0, dt], state, args=(Tc, params),
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y[:, -1]


def simulate_cstr_trajectory(initial_state, Tc_sequence, dt=DT, params=None):
    N = len(Tc_sequence)
    states = np.zeros((N + 1, 2))
    states[0] = initial_state
    for k in range(N):
        states[k + 1] = simulate_cstr_step(states[k], Tc_sequence[k], dt, params)
        states[k + 1, 0] = np.clip(states[k + 1, 0], 0.0, 2.0)
        states[k + 1, 1] = np.clip(states[k + 1, 1], 250.0, 500.0)
    return states


def steady_state_from_T(T_ss, params=None):
    """
    Reachable steady state of the single-input plant for a given reactor
    temperature T_ss. Returns (CA_ss, Tc_ss). With one manipulated input,
    only pairs (CA, T) on this one-dimensional manifold can be held at
    steady state; arbitrary independent CA/T setpoints are generally
    unreachable.
    """
    params = CSTR_PARAMS if params is None else params
    q, V = params['q'], params['V']
    k = params['k0'] * np.exp(-params['Ea_over_R'] / T_ss)
    CA_ss = (q / V) * params['CA_f'] / (q / V + k)
    rA = k * CA_ss
    Tc_ss = T_ss - ((q / V) * (params['T_f'] - T_ss)
                    + (-params['delta_H'] / params['rho_Cp']) * rA) \
        / (params['UA'] / (params['rho_Cp'] * V))
    return CA_ss, Tc_ss
