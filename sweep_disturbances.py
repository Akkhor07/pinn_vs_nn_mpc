"""
sweep_disturbances.py — Disturbance-magnitude sweep (robustness envelope).

Per seed (uses the models trained by run_experiments.py, no retraining):
    python sweep_disturbances.py results/seed_45
Across all seeds that have a sweep.json:
    python sweep_disturbances.py --aggregate results
For every disturbance type and size in config.SWEEP_GRID, each controller runs the
standard step-reference closed loop against the disturbed plant (step applied at t=0).
"""
import glob
import json
import os
import sys
import time
import numpy as np
import torch
from config import CSTR_PARAMS, SWEEP_GRID, CONTROLLERS, config_snapshot
from models import BlackBoxNN, PhysicsInformedNN
from mpc_controller import NeuralMPC, FirstPrinciplesMPC
from closed_loop_sim import run_closed_loop

KEEP = ['rmse_T', 'offset_T', 'max_T', 'ignited', 'n_failed_solves', 'solve_ms_mean', 'control_effort']
UNITS = {'CA_f': '% increase in $C_{A,f}$', 'T_f': 'K increase in $T_f$', 'k0': '% increase in $k_0$'}


def plant_params(kind, mag):
    p = dict(CSTR_PARAMS)
    if kind == 'CA_f':
        p['CA_f_disturbed'] = CSTR_PARAMS['CA_f'] * (1 + mag / 100)
    elif kind == 'T_f':
        p['T_f_disturbed'] = CSTR_PARAMS['T_f'] + mag
    elif kind == 'k0':
        p['k0_true'] = CSTR_PARAMS['k0'] * (1 + mag / 100)
    return p


def load_controllers(seed_dir):
    ns = dict(np.load(os.path.join(seed_dir, 'norm_stats.npz')))
    bb = BlackBoxNN(); bb.load_state_dict(torch.load(os.path.join(seed_dir, 'blackbox_model.pt')))
    pinn = PhysicsInformedNN(norm_stats=ns)
    pinn.load_state_dict(torch.load(os.path.join(seed_dir, 'pinn_model.pt')))
    build = {'blackbox': lambda: NeuralMPC(bb, ns, 'blackbox'),
             'pinn': lambda: NeuralMPC(pinn, ns, 'pinn'),
             'first_principles': lambda: FirstPrinciplesMPC()}
    return {n: build[n]() for n in CONTROLLERS}


def run_seed(seed_dir):
    ctrls = load_controllers(seed_dir)
    out = {'config': config_snapshot(), 'results': {}}
    t0 = time.time()
    nominal_cache = {}   # zero disturbance is identical for every kind: run once per controller
    for kind, mags in SWEEP_GRID.items():
        out['results'][kind] = {}
        for mag in mags:
            out['results'][kind][str(mag)] = {}
            for name, c in ctrls.items():
                if mag == 0 and name in nominal_cache:
                    r = nominal_cache[name]
                else:
                    r = run_closed_loop(c, 'step', plant_params=plant_params(kind, mag))
                    if mag == 0:
                        nominal_cache[name] = r
                out['results'][kind][str(mag)][name] = {k: r[k] for k in KEEP}
                print(f"  {kind}+{mag:<3} {name:<17} RMSE_T={r['rmse_T']:6.2f} offset_T={r['offset_T']:5.2f} "
                      f"maxT={r['max_T']:5.0f} ignited={r['ignited']}  [{time.time()-t0:5.0f}s]", flush=True)
    with open(os.path.join(seed_dir, 'sweep.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {seed_dir}/sweep.json")


def aggregate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(root, 'seed_*', 'sweep.json')))
    if not files:
        sys.exit("no sweep.json files found")
    S = [json.load(open(f))['results'] for f in files]
    style = {'blackbox': ('#1f77b4', 'o', 'Black-box NN MPC'), 'pinn': ('#d62728', 's', 'PINN MPC'),
             'first_principles': ('0.3', '^', 'First-principles NMPC')}
    kinds = list(S[0].keys())
    out = os.path.join(root, 'aggregate'); os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(2, len(kinds), figsize=(5 * len(kinds), 7), squeeze=False)
    rows = ['kind,magnitude,controller,n_seeds,rmse_T_mean,rmse_T_sd,ignition_frac']
    for j, kind in enumerate(kinds):
        mags = sorted(S[0][kind].keys(), key=float)
        x = np.array([float(m) for m in mags])
        for name in S[0][kind][mags[0]]:
            col, mk, lab = style[name]
            R = np.array([[s[kind][m][name]['rmse_T'] for m in mags] for s in S])
            I = np.array([[s[kind][m][name]['ignited'] for m in mags] for s in S], float)
            mu, sd = R.mean(0), R.std(0, ddof=1) if len(S) > 1 else np.zeros(len(mags))
            axes[0, j].errorbar(x, mu, yerr=sd, color=col, marker=mk, capsize=3, label=lab)
            ls = {'blackbox': '-', 'pinn': '--', 'first_principles': ':'}[name]
            axes[1, j].plot(x, I.mean(0), color=col, marker=mk, ls=ls, mfc='none', ms=8, label=lab)
            for m, a, b, c in zip(mags, mu, sd, I.mean(0)):
                rows.append(f"{kind},{m},{name},{len(S)},{a:.4f},{b:.4f},{c:.3f}")
        axes[0, j].set_title(f"({'abc'[j]}) {kind} disturbance"); axes[0, j].set_ylabel(r'RMSE$_T$ (K)')
        axes[1, j].set_ylabel('Fraction of seeds ignited'); axes[1, j].set_ylim(-0.05, 1.05)
        for a in axes[:, j]:
            a.set_xlabel(UNITS[kind]); a.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Disturbance-magnitude sweep (mean ± SD over {len(S)} seeds)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_sweep.{ext}'), dpi=300, bbox_inches='tight')
    with open(os.path.join(out, 'sweep_summary.csv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"Aggregated {len(S)} seeds -> {out}/fig_sweep.png, sweep_summary.csv")


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--aggregate':
        aggregate(sys.argv[2])
    else:
        run_seed(sys.argv[1] if len(sys.argv) > 1 else 'results/seed_45')
