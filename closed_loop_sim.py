"""
closed_loop_sim.py — Closed-loop simulation of any controller against the
high-fidelity plant. No online model adaptation (models are frozen).
"""
import numpy as np
from config import (CL_SIM_STEPS, DT, SS_CA, SS_T, SS_TC, CSTR_PARAMS,
                    REFERENCE_MODE, SETPOINT_T_LEVELS, SETPOINT_SWITCH_FRACS,
                    LEGACY_CA_LEVELS, STEADY_WINDOW_FRAC, DISTURBANCE_SCENARIOS,
                    DISTURBANCE_ONSET_STEP, CONTROL_MODE, IGNITION_T)
from cstr_plant import simulate_cstr_step, steady_state_from_T
from mpc_controller import SUCCESS
from estimator import DisturbanceEKF


def switch_steps(n_steps):
    return [int(round(f * n_steps)) for f in SETPOINT_SWITCH_FRACS]


def generate_reference_trajectory(profile_name='step', n_steps=CL_SIM_STEPS):
    """
    Returns (ref, u_ss): ref is (n_steps, 2) [CA_sp, T_sp]; u_ss is (n_steps,)
    the steady-state coolant temperature for each setpoint (nominal model).
    'reachable': CA_sp is the steady state consistent with T_sp for the
    single-input plant, so every setpoint pair can actually be held.
    """
    if profile_name != 'step':
        raise ValueError(f"Unknown profile: {profile_name}")
    sw = switch_steps(n_steps) + [n_steps]
    ref = np.zeros((n_steps, 2)); u_ss = np.zeros(n_steps)
    for i, T_sp in enumerate(SETPOINT_T_LEVELS):
        CA_ss, Tc_ss = steady_state_from_T(T_sp, CSTR_PARAMS)
        CA_sp = CA_ss if REFERENCE_MODE == 'reachable' else LEGACY_CA_LEVELS[i]
        ref[sw[i]:sw[i + 1]] = [CA_sp, T_sp]
        u_ss[sw[i]:sw[i + 1]] = Tc_ss
    return ref, u_ss


def _horizon(arr, k, N):
    h = arr[k:k + N]
    if len(h) < N:
        h = np.concatenate([h, np.repeat(h[-1:], N - len(h), axis=0)], axis=0)
    return h


def run_closed_loop(controller, profile_name='step', disturbance_name='none',
                    n_steps=CL_SIM_STEPS, initial_state=None, plant_params=None):
    """
    plant_params: full plant parameter dict (overrides disturbance_name), used by Monte Carlo.
    """
    x0 = np.array([SS_CA, SS_T]) if initial_state is None else np.asarray(initial_state, float)
    ref, u_ss = generate_reference_trajectory(profile_name, n_steps)
    if plant_params is None:
        plant_params = dict(CSTR_PARAMS); plant_params.update(DISTURBANCE_SCENARIOS[disturbance_name])
    nominal = dict(CSTR_PARAMS)

    N = controller.N
    controller.reset(x0)
    of = CONTROL_MODE == 'offset_free'
    if of:
        if getattr(controller, 'ekf', None) is None:
            controller.ekf = DisturbanceEKF(controller.f_pred)
        controller.ekf.reset(x0)
    states = np.zeros((n_steps + 1, 2)); states[0] = x0
    inputs = np.zeros(n_steps); statuses = []
    d_hat = np.zeros(2); d_hist = np.zeros((n_steps, 2))
    Tc_prev = SS_TC
    targets, tgt_res = {}, []
    for k in range(n_steps):
        x_meas = states[k]
        if of and k > 0:
            x_meas, d_hat = controller.ekf.step(Tc_prev, states[k])
        ref_h, u_h = _horizon(ref, k, N).copy(), _horizon(u_ss, k, N).copy()
        if of:   # replace (CA_sp, Tc_ss) by the disturbance-consistent target for each T_sp
            for T_sp in np.unique(ref_h[:, 1]):
                guess = targets.get(T_sp, (ref_h[ref_h[:, 1] == T_sp][0, 0], u_h[ref_h[:, 1] == T_sp][0]))
                CA_s, Tc_s, res = controller.target_calc(T_sp, d_hat, guess)
                targets[T_sp] = (CA_s, Tc_s); tgt_res.append(res)
                mask = ref_h[:, 1] == T_sp
                ref_h[mask, 0], u_h[mask] = CA_s, Tc_s
        Tc, st = controller.solve(x_meas, ref_h, Tc_prev, u_h, d_hat if of else (0.0, 0.0))
        inputs[k] = Tc; statuses.append(st); d_hist[k] = d_hat
        p = plant_params if k >= DISTURBANCE_ONSET_STEP else nominal
        states[k + 1] = simulate_cstr_step(states[k], Tc, DT, p)
        states[k + 1] = [np.clip(states[k + 1, 0], 0.0, 1.5), np.clip(states[k + 1, 1], 250.0, 500.0)]
        Tc_prev = Tc

    e = states[1:] - ref
    ignited = bool(np.max(states[:, 1]) > IGNITION_T)
    # steady-state offset: mean |error| over the last STEADY_WINDOW_FRAC of each segment
    sw = switch_steps(n_steps) + [n_steps]
    off_CA, off_T = [], []
    for i in range(len(SETPOINT_T_LEVELS)):
        a, b = sw[i], sw[i + 1]
        w = max(3, int(round(STEADY_WINDOW_FRAC * (b - a))))
        off_CA.append(float(np.mean(np.abs(e[b - w:b, 0]))))
        off_T.append(float(np.mean(np.abs(e[b - w:b, 1]))))
    st = np.array(controller.solve_times)
    n_failed = sum(1 for s in statuses if s not in SUCCESS)

    return {
        'time': np.arange(n_steps + 1) * DT, 'states': states, 'inputs': inputs,
        'reference': ref, 'u_ss': u_ss, 'd_hat': d_hist,
        'rmse_CA': float(np.sqrt(np.mean(e[:, 0] ** 2))),
        'rmse_T': float(np.sqrt(np.mean(e[:, 1] ** 2))),
        'mae_CA': float(np.mean(np.abs(e[:, 0]))), 'mae_T': float(np.mean(np.abs(e[:, 1]))),
        'max_err_T': float(np.max(np.abs(e[:, 1]))),
        'offset_CA': float(np.mean(off_CA)), 'offset_T': float(np.mean(off_T)),
        'offset_T_per_segment': off_T,
        'control_effort': float(np.sum(np.abs(np.diff(inputs)))),
        'n_failed_solves': int(n_failed),
        'target_unreachable_frac': float(np.mean(np.array(tgt_res) > 1e-3)) if tgt_res else 0.0, 'ignited': ignited, 'max_T': float(np.max(states[:, 1])),
        'solve_ms_mean': float(1e3 * st.mean()), 'solve_ms_max': float(1e3 * st.max()),
    }
