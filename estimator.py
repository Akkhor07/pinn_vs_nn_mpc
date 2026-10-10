"""
estimator.py — EKF for offset-free MPC (augmented constant-disturbance model).

    x_{k+1} = f_hat(x_k, u_k) + d_k,     d_{k+1} = d_k + w_d,     y_k = x_k + v_k
State z = [CA, T, d_CA, d_T]. Both states are measured (C = [I 0]); the
augmented system is detectable because rank([I-A, -I; I, 0]) = 4.
"""
import numpy as np
import casadi as ca
from config import EKF_Q_X, EKF_Q_D, EKF_R


class DisturbanceEKF:
    def __init__(self, f_pred):
        z, u = ca.SX.sym('z', 4), ca.SX.sym('u')
        CAn, Tn = f_pred(z[0], z[1], u)
        fz = ca.vertcat(CAn + z[2], Tn + z[3], z[2], z[3])
        self.F = ca.Function('F', [z, u], [fz, ca.jacobian(fz, z)])
        self.Q = np.diag([EKF_Q_X[0], EKF_Q_X[1], EKF_Q_D[0], EKF_Q_D[1]])
        self.R = np.diag(EKF_R)
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])
        self.reset(np.array([0.5, 350.0]))

    def reset(self, y0):
        self.z = np.array([y0[0], y0[1], 0.0, 0.0])
        self.P = np.diag([EKF_R[0], EKF_R[1], 10 * EKF_Q_D[0], 10 * EKF_Q_D[1]])

    def step(self, u_prev, y):
        """Predict with the input applied over the last interval, then correct with y."""
        zp, A = self.F(self.z, u_prev)
        zp, A = np.array(zp).ravel(), np.array(A)
        Pp = A @ self.P @ A.T + self.Q
        S = self.H @ Pp @ self.H.T + self.R
        K = Pp @ self.H.T @ np.linalg.inv(S)
        self.z = zp + K @ (np.asarray(y) - zp[:2])
        self.P = (np.eye(4) - K @ self.H) @ Pp
        return self.z[:2].copy(), self.z[2:].copy()
