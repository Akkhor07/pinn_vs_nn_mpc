"""
mpc_controller.py — Multiple-shooting NMPC (CasADi/IPOPT) with interchangeable
prediction models:
    NeuralMPC          : black-box NN or PINN (weights re-embedded symbolically)
    FirstPrinciplesMPC : the physics prior itself, integrated with RK4 sub-steps

CONTROL_MODE='offset_free' (default), stage k = 0..N-1:
    q_T (T_{k+1}-T_sp)^2 + q_CA (CA_{k+1}-CA_s,k)^2 + r (Tc_k-Tc_s,k)^2 + r_d (Tc_k-Tc_{k-1})^2
    with (CA_s,k, Tc_s,k) the disturbance-consistent steady state holding T_sp.
CONTROL_MODE='nominal' (ablation): q_CA (CA-CA_sp)^2 + q_T (T-T_sp)^2 + r (Tc-Tc_ss)^2 + r_d dTc^2.
Prediction: x_{k+1} = f_hat(x_k, Tc_k) + d_hat, d_hat from the EKF (zero in nominal mode).
"""
import time
import numpy as np
import casadi as ca
from config import (MPC_HORIZON, Q_WEIGHT, R_WEIGHT, R_DELTA_WEIGHT, INPUT_PENALTY,
                    TC_MIN, TC_MAX, CA_MIN, CA_MAX, T_MIN, T_MAX, SS_TC,
                    IPOPT_MAX_ITER, IPOPT_TOL, FP_SUBSTEPS, DT, physics_prior_params,
                    CONTROL_MODE, Q_T_OF, Q_CA_OF, RHS_T_CLAMP)
CONTROL_Mode_OF = CONTROL_MODE == 'offset_free'

# 'Search_Direction_Becomes_Too_Small': IPOPT cannot make further progress; iterate is converged within tolerance
SUCCESS = ('Solve_Succeeded', 'Solved_To_Acceptable_Level', 'Search_Direction_Becomes_Too_Small')


class _BaseMPC:
    model_type = 'base'

    def __init__(self):
        self.N = MPC_HORIZON
        CA, T, Tc = ca.SX.sym('CA'), ca.SX.sym('T'), ca.SX.sym('Tc')
        CAn, Tn = self._predict_sym(CA, T, Tc)
        self.f_pred = ca.Function('f_pred', [CA, T, Tc], [CAn, Tn])
        self._build_nlp()
        self.reset()

    # subclasses implement the one-step symbolic model
    def _predict_sym(self, CA, T, Tc):
        raise NotImplementedError

    def predict(self, state, Tc):
        CAn, Tn = self.f_pred(state[0], state[1], Tc)
        return np.array([float(CAn), float(Tn)])

    def _build_nlp(self):
        """
        Dynamic NLP (same structure in both modes).
        w = [CA_0..N, T_0..N, Tc_0..N-1]
        p = [CA0, T0, (CA_tgt,T_sp) x N, Tc_prev, Tc_tgt x N, d_CA, d_T]
        nominal mode    : CA_tgt = CA_sp (reachable nominal), Tc_tgt = nominal Tc_ss, d = 0
        offset-free mode: (CA_tgt, Tc_tgt) from target_calc() under the current d_hat
        """
        N = self.N
        w = ca.SX.sym('w', 2 * (N + 1) + N)
        p = ca.SX.sym('p', 2 + 2 * N + 1 + N + 2)
        CA = [w[i] for i in range(N + 1)]
        T = [w[N + 1 + i] for i in range(N + 1)]
        Tc = [w[2 * (N + 1) + i] for i in range(N)]
        CA0, T0 = p[0], p[1]
        CAr = [p[2 + 2 * i] for i in range(N)]
        Tr = [p[3 + 2 * i] for i in range(N)]
        Tc_prev = p[2 + 2 * N]
        u_t = [p[3 + 2 * N + i] for i in range(N)]
        dCA, dT = p[3 + 3 * N], p[4 + 3 * N]
        of = CONTROL_Mode_OF
        qCA, qT = (Q_CA_OF, Q_T_OF) if of else (Q_WEIGHT[0, 0], Q_WEIGHT[1, 1])

        J, g = 0, [CA[0] - CA0, T[0] - T0]
        for k in range(N):
            J += qCA * (CA[k + 1] - CAr[k]) ** 2 + qT * (T[k + 1] - Tr[k]) ** 2
            J += R_WEIGHT[0, 0] * ((Tc[k] - u_t[k]) if (of or INPUT_PENALTY == 'deviation') else Tc[k]) ** 2
            J += R_DELTA_WEIGHT[0, 0] * (Tc[k] - (Tc_prev if k == 0 else Tc[k - 1])) ** 2
            CAn, Tn = self.f_pred(CA[k], T[k], Tc[k])
            g += [CA[k + 1] - (CAn + dCA), T[k + 1] - (Tn + dT)]

        self.lbw = [CA_MIN] * (N + 1) + [T_MIN] * (N + 1) + [TC_MIN] * N
        self.ubw = [CA_MAX] * (N + 1) + [T_MAX] * (N + 1) + [TC_MAX] * N
        self.lbg = self.ubg = [0.0] * len(g)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
                'ipopt.max_iter': IPOPT_MAX_ITER, 'ipopt.tol': IPOPT_TOL}
        self.solver = ca.nlpsol('solver', 'ipopt',
                                {'x': w, 'f': J, 'g': ca.vertcat(*g), 'p': p}, opts)

        # ---- steady-state target calculation (offset-free): 2 unknowns ----
        # find (CA_s, Tc_s) with  CA_s = f_CA(CA_s,T_sp,Tc_s) + d_CA,  T_sp = f_T(CA_s,T_sp,Tc_s) + d_T
        # posed as bounded least squares so it is always solvable; exact when reachable.
        z, q = ca.SX.sym('z', 2), ca.SX.sym('q', 3)          # z=[CA_s,Tc_s], q=[T_sp,d_CA,d_T]
        CAt, Tt = self.f_pred(z[0], q[0], z[1])
        res = ca.vertcat((CAt + q[1] - z[0]) / 0.01, Tt + q[2] - q[0])
        self.target_solver = ca.nlpsol('target', 'ipopt', {'x': z, 'p': q, 'f': ca.sumsqr(res)},
                                       {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
                                        'ipopt.max_iter': 100, 'ipopt.tol': 1e-10})
        self.target_res = ca.Function('target_res', [z, q], [res])

    def target_calc(self, T_sp, d, guess):
        """Returns (CA_s, Tc_s, residual_norm). Residual > 0 means target unreachable (best compromise)."""
        sol = self.target_solver(x0=guess, p=[T_sp, d[0], d[1]],
                                 lbx=[CA_MIN, TC_MIN], ubx=[CA_MAX, TC_MAX])
        z = np.array(sol['x']).ravel()
        r = float(np.linalg.norm(np.array(self.target_res(z, [T_sp, d[0], d[1]])).ravel()))
        return float(z[0]), float(z[1]), r

    def reset(self, state=None):
        """Clear warm start (call at the start of every closed-loop run)."""
        CA0, T0 = (0.5, 350.0) if state is None else (state[0], state[1])
        N = self.N
        self.w0 = [CA0] * (N + 1) + [T0] * (N + 1) + [SS_TC] * N
        self.solve_times = []

    def solve(self, state, reference, Tc_prev, u_ss=None, d_hat=(0.0, 0.0)):
        """Returns (Tc_opt, status). reference: (N,2) [CA_sp, T_sp]; u_ss: (N,) Tc_ss."""
        N = self.N
        u_ss = np.full(N, SS_TC) if u_ss is None else np.asarray(u_ss, float)
        pv = [state[0], state[1]] + list(np.asarray(reference)[:N].ravel()) + \
            [Tc_prev] + list(u_ss[:N]) + [d_hat[0], d_hat[1]]
        t0 = time.perf_counter()
        try:
            sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw,
                              lbg=self.lbg, ubg=self.ubg, p=pv)
            status = self.solver.stats()['return_status']
            w = np.array(sol['x']).ravel()
            self.w0 = list(w)
            Tc_opt = float(w[2 * (N + 1)])
        except Exception as e:  # noqa: BLE001
            status, Tc_opt = f'exception: {e}', float(Tc_prev)
        self.solve_times.append(time.perf_counter() - t0)
        return Tc_opt, status


class NeuralMPC(_BaseMPC):
    def __init__(self, model, norm_stats, model_type='blackbox'):
        self.model_type = model_type
        self.norm = norm_stats
        self.weights, self.biases = [], []
        for layer in model.net:
            if hasattr(layer, 'weight'):
                self.weights.append(layer.weight.detach().cpu().numpy().astype(float))
                self.biases.append(layer.bias.detach().cpu().numpy().astype(float))
        super().__init__()

    def _predict_sym(self, CA, T, Tc):
        n = self.norm
        h = (ca.vertcat(CA, T, Tc) - n['X_mean']) / n['X_std']
        for i in range(len(self.weights)):
            h = ca.mtimes(ca.DM(self.weights[i]), h) + ca.DM(self.biases[i])
            if i < len(self.weights) - 1:
                h = ca.tanh(h)
        y = h * n['Y_std'] + n['Y_mean']
        return y[0], y[1]


class FirstPrinciplesMPC(_BaseMPC):
    """NMPC using the physics prior (nominal x PHYSICS_PARAM_MISMATCH) as model.
    With no mismatch this is the exact plant model -> performance upper bound."""
    model_type = 'first_principles'

    def __init__(self, params=None, n_sub=FP_SUBSTEPS):
        self.params = physics_prior_params() if params is None else params
        self.n_sub = n_sub
        super().__init__()

    def _predict_sym(self, CA, T, Tc):
        P = self.params
        lo, hi = RHS_T_CLAMP
        def rhs(c, t):
            # numerical-safety bounds only: keep exp() finite when IPOPT explores
            # extreme iterates (inactive along physical trajectories)
            t = ca.fmin(ca.fmax(t, lo), hi); c = ca.fmax(c, 0.0)
            r = P['k0'] * ca.exp(-P['Ea_over_R'] / t) * c
            return ((P['q'] / P['V']) * (P['CA_f'] - c) - r,
                    (P['q'] / P['V']) * (P['T_f'] - t) + (-P['delta_H'] / P['rho_Cp']) * r
                    + (P['UA'] / (P['rho_Cp'] * P['V'])) * (Tc - t))
        h = DT / self.n_sub
        c, t = CA, T
        for _ in range(self.n_sub):
            k1 = rhs(c, t)
            k2 = rhs(c + 0.5 * h * k1[0], t + 0.5 * h * k1[1])
            k3 = rhs(c + 0.5 * h * k2[0], t + 0.5 * h * k2[1])
            k4 = rhs(c + h * k3[0], t + h * k3[1])
            c = c + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            t = t + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return c, t
