"""
config.py — Central configuration. EVERY number used by the pipeline and
reported in the paper should live here (single source of truth).

Environment overrides (so seeds can be swept without editing this file):
    PINN_SEED=<int>   random seed for data, training, Monte Carlo
    PINN_SMOKE=1      tiny settings for a fast end-to-end code check (NOT for results)
"""
import os
import numpy as np

# ---------------------------------------------------------------- run control
RANDOM_SEED = int(os.environ.get('PINN_SEED', 45))
SMOKE = os.environ.get('PINN_SMOKE', '0') == '1'
RESULTS_ROOT = 'results'

# ------------------------------------------------------ CSTR physical parameters
CSTR_PARAMS = {
    'q': 100.0,          # L/min
    'V': 100.0,          # L
    'CA_f': 1.0,         # mol/L
    'T_f': 350.0,        # K
    'k0': 7.2e10,        # 1/min
    'Ea_over_R': 8750.0, # K
    'delta_H': -5.0e4,   # J/mol
    'rho_Cp': 239.0,     # J/(L K)
    'UA': 5.0e4,         # J/(min K)
}

# Nominal steady state (consistent: Tc_ss is computed from the model for T=350 K)
SS_CA = 0.5
SS_T = 350.0
SS_TC = 300.0

# State / input bounds used by the MPC
CA_MIN, CA_MAX = 0.0, 1.0
T_MIN, T_MAX = 300.0, 450.0
TC_MIN, TC_MAX = 250.0, 350.0

# ------------------------------------------------------------ data generation
DT = 0.5                      # min
N_TRAIN_TRAJECTORIES = 200
N_VAL_TRAJECTORIES = 50
N_TEST_TRAJECTORIES = 50
TRAJ_LENGTH = 50
NOISE_STD_CA = 0.005
NOISE_STD_T = 0.5
TC_STEP_MIN = 270.0
TC_STEP_MAX = 330.0
TC_CHANGE_PROB = 0.1

# --------------------------------------------------------------- networks
HIDDEN_DIM = 64
N_HIDDEN_LAYERS = 3
LEARNING_RATE = 1e-3
N_EPOCHS_BLACKBOX = 500
N_EPOCHS_PINN = 500
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-5

# ----------------------------------------------------------- physics residual
LAMBDA_PHYSICS = 0.02         # lambda_max
LAMBDA_WARMUP_FRAC = 0.6      # fraction of epochs held at lambda_max
LAMBDA_MIN_FRAC = 0.05        # final lambda = lambda_max * LAMBDA_MIN_FRAC
N_COLLOCATION = 500           # collocation points per mini-batch

# 'precomputed': physics targets Phi(x,u) computed ONCE offline with a
#                high-accuracy integrator (validated vs. scipy RK45 at
#                rtol=1e-8). Exact up to integrator tolerance; fast training.
# 'rk4_online' : legacy mode, explicit RK4 inside the loss (needed only if
#                LEARN_K0=True). Requires ~320 sub-steps to be accurate over
#                the collocation domain because of ignition transients.
PHYSICS_TARGET_MODE = 'precomputed'
PHYSICS_TARGET_SUBSTEPS = 2000  # RK4 sub-steps for offline precomputation (max |eT| ~0.02 K vs scipy RK45)
N_COLLOCATION_POOL = 20000      # size of the fixed collocation pool
RK4_ONLINE_SUBSTEPS = 320
LEARN_K0 = False                # only meaningful with 'rk4_online'
RHS_T_CLAMP = (250.0, 800.0)    # numerical-safety clamp only (old 450 K clamp truncated real hot spots)
COLLOCATION_RANGES = {'CA': (0.2, 0.8), 'T': (320.0, 380.0), 'Tc': (270.0, 330.0)}

# Deliberate physics mismatch (review comment M2). Multiplicative factors
# applied to the physics PRIOR only (PINN residual and first-principles NMPC
# model). The simulated plant always uses CSTR_PARAMS. Empty dict = exact model.
#   e.g. PHYSICS_PARAM_MISMATCH = {'UA': 0.8}
PHYSICS_PARAM_MISMATCH = {}

# -------------------------------------------------------------------- MPC
MPC_HORIZON = 10
MPC_DT = DT
Q_WEIGHT = np.diag([10.0, 1.0])
R_WEIGHT = np.array([[0.01]])
R_DELTA_WEIGHT = np.array([[1.0]])
# 'deviation': penalize (Tc - Tc_ss(ref))^2  (fix for review comment M1)
# 'absolute' : penalize Tc^2 (old behaviour; Tc ~ 300 K made this term dominate)
INPUT_PENALTY = 'deviation'
# Controller formulation
# 'offset_free': augmented-disturbance EKF (estimator.py) + separate steady-state
#                target calculation (2-variable problem) before each dynamic NLP. T is the controlled variable; the CA
#                target is the steady state consistent with T_sp under the
#                current disturbance estimate (1 input -> 1 offset-free output).
# 'nominal'    : no disturbance estimation; tracks the (CA_sp, T_sp) pair
#                with the nominal model (earlier formulation, kept as ablation).
CONTROL_MODE = 'offset_free'
Q_T_OF = 1.0          # weight on (T - T_sp)^2 in offset-free mode
Q_CA_OF = 10.0        # weight on (CA - CA_s)^2, CA_s = model-consistent target
EKF_Q_X = (1e-6, 1e-2)     # process noise variance on (CA, T)
EKF_Q_D = (1e-6, 1e-2)     # random-walk variance on (d_CA, d_T)
EKF_R = (1e-6, 1e-2)       # measurement noise variance on (CA, T)
FP_SUBSTEPS = 200               # RK4 sub-steps in the first-principles NMPC model (max |eT| ~0.014 K vs scipy)
IPOPT_MAX_ITER = 200
IPOPT_TOL = 1e-6
CONTROLLERS = ['blackbox', 'pinn', 'first_principles']

# ------------------------------------------------------------ closed loop
CL_SIM_STEPS = 200
# Reference: temperature levels; CA is computed so every (CA, T) pair is a
# reachable steady state of the single-input plant (fix for review comment M1).
# 'legacy' reproduces the old, partly unreachable CA profile.
REFERENCE_MODE = 'reachable'
SETPOINT_T_LEVELS = [350.0, 360.0, 370.0, 360.0, 350.0]
SETPOINT_SWITCH_FRACS = [0.0, 0.25, 0.5, 0.75, 0.875]
LEGACY_CA_LEVELS = [0.5, 0.4, 0.3, 0.4, 0.5]
STEADY_WINDOW_FRAC = 0.2        # last 20% of each reference segment used for offset metric

# Disturbances are step changes in the PLANT parameters applied from
# DISTURBANCE_ONSET_STEP onward. The controller's model is never told.
DISTURBANCE_ONSET_STEP = 0
DISTURBANCE_SCENARIOS = {
    'none': {},
    'feed_concentration': {'CA_f_disturbed': 1.1},
    'feed_temperature': {'T_f_disturbed': 355.0},
    'parameter_mismatch': {'k0_true': 7.5e10},
    'combined': {'CA_f_disturbed': 1.05, 'T_f_disturbed': 352.0, 'k0_true': 7.4e10},
}

# ------------------------------------------------- disturbance magnitude sweep
# sweep_disturbances.py: step disturbance of each type and size, all controllers.
SWEEP_GRID = {
    'CA_f': [0, 2, 4, 6, 8, 10],      # % increase in feed concentration
    'T_f':  [0, 1, 2, 3, 4, 5],       # K increase in feed temperature
    'k0':   [0, 1, 2, 4, 6, 8],       # % increase in pre-exponential factor
}
IGNITION_T = 400.0                    # K; a run with max T above this counts as ignited

# ------------------------------------------------------------ Monte Carlo
N_MONTE_CARLO = 30
# Per trial, independently: CA_f ~ N(1.0, (0.05*1.0)^2), T_f ~ N(350, 3^2),
# k0 ~ 7.2e10 * (1 + N(0, 0.03^2)), x0 = (0.5 + N(0,0.02^2), 350 + N(0,2^2)).
MC_SIGMA = {'CA_f_rel': 0.05, 'T_f_abs': 3.0, 'k0_rel': 0.03, 'CA0_abs': 0.02, 'T0_abs': 2.0}

# ------------------------------------------------------------ smoke mode
if SMOKE:
    N_TRAIN_TRAJECTORIES, N_VAL_TRAJECTORIES, N_TEST_TRAJECTORIES = 20, 5, 5
    N_EPOCHS_BLACKBOX = N_EPOCHS_PINN = 4
    N_COLLOCATION_POOL = 2000
    CL_SIM_STEPS = 40
    N_MONTE_CARLO = 2
    SWEEP_GRID = {'CA_f': [0, 6], 'T_f': [3], 'k0': [4]}
    RESULTS_ROOT = 'results_smoke'


def physics_prior_params():
    """Parameters the PHYSICS PRIOR believes in (nominal x mismatch factors)."""
    p = dict(CSTR_PARAMS)
    for k, f in PHYSICS_PARAM_MISMATCH.items():
        p[k] = p[k] * f
    return p


def config_snapshot():
    """Serializable record of the settings used for a run (saved with results)."""
    keys = [k for k in globals() if k.isupper()]
    out = {}
    for k in keys:
        v = globals()[k]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out
