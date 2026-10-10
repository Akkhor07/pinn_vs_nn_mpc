"""
run_experiments.py — Full pipeline for ONE seed (set PINN_SEED env var or config).
Writes everything to <RESULTS_ROOT>/seed_<seed>/ :
    summary.json      all scalar metrics + config snapshot (input to aggregate_seeds.py)
    trajectories.npz  closed-loop time series for plotting
    figures (png/pdf)
"""
import json
import os
import time
import numpy as np
import torch
from config import (RANDOM_SEED, RESULTS_ROOT, N_MONTE_CARLO, MC_SIGMA, CSTR_PARAMS,
                    DISTURBANCE_SCENARIOS, CONTROLLERS, SS_CA, SS_T, config_snapshot)
from data_generation import (generate_all_datasets, generate_random_initial_state,
                             generate_random_input_sequence)
from train_blackbox import train_blackbox
from train_pinn import train_pinn
from mpc_controller import NeuralMPC, FirstPrinciplesMPC
from closed_loop_sim import run_closed_loop
from cstr_plant import simulate_cstr_trajectory

SCALARS = ['rmse_CA', 'rmse_T', 'mae_CA', 'mae_T', 'max_err_T', 'offset_CA', 'offset_T',
           'control_effort', 'n_failed_solves', 'solve_ms_mean', 'solve_ms_max',
           'ignited', 'max_T', 'target_unreachable_frac']


def compute_open_loop_metrics(model, data, n_rollout=20, n_trajs=50):
    norm = data['norm_stats']
    model.eval()
    with torch.no_grad():
        Yn = model(torch.FloatTensor((data['X_test'] - norm['X_mean']) / norm['X_std'])).numpy()
    err = Yn * norm['Y_std'] + norm['Y_mean'] - data['Y_test']
    one = {'rmse_CA': float(np.sqrt(np.mean(err[:, 0] ** 2))), 'rmse_T': float(np.sqrt(np.mean(err[:, 1] ** 2))),
           'mae_CA': float(np.mean(np.abs(err[:, 0]))), 'mae_T': float(np.mean(np.abs(err[:, 1])))}
    rng = np.random.default_rng(RANDOM_SEED + 999)
    eCA, eT = [], []
    for _ in range(n_trajs):
        x0 = generate_random_initial_state(rng)
        u = generate_random_input_sequence(n_rollout, rng)
        truth = simulate_cstr_trajectory(x0, u)
        pred = np.zeros((n_rollout + 1, 2)); pred[0] = x0
        for k in range(n_rollout):
            pred[k + 1] = model.predict_denormalized(pred[k, 0], pred[k, 1], u[k], norm)
        eCA.append(np.abs(pred[1:, 0] - truth[1:, 0])); eT.append(np.abs(pred[1:, 1] - truth[1:, 1]))
    eCA, eT = np.array(eCA), np.array(eT)
    roll = {'mean_CA': eCA.mean(0).tolist(), 'std_CA': eCA.std(0).tolist(),
            'mean_T': eT.mean(0).tolist(), 'std_T': eT.std(0).tolist()}
    return {'one_step': one, 'rollout': roll}


def monte_carlo(controllers):
    """Per trial: perturbed plant (CA_f, T_f, k0) and initial state; same draws for every controller."""
    rng = np.random.default_rng(RANDOM_SEED + 100)
    out = {name: [] for name in controllers}
    for trial in range(N_MONTE_CARLO):
        p = dict(CSTR_PARAMS)
        p['CA_f_disturbed'] = CSTR_PARAMS['CA_f'] * (1 + rng.normal(0, MC_SIGMA['CA_f_rel']))
        p['T_f_disturbed'] = CSTR_PARAMS['T_f'] + rng.normal(0, MC_SIGMA['T_f_abs'])
        p['k0_true'] = CSTR_PARAMS['k0'] * (1 + rng.normal(0, MC_SIGMA['k0_rel']))
        x0 = [SS_CA + rng.normal(0, MC_SIGMA['CA0_abs']), SS_T + rng.normal(0, MC_SIGMA['T0_abs'])]
        for name, ctrl in controllers.items():
            r = run_closed_loop(ctrl, 'step', initial_state=x0, plant_params=p)
            out[name].append({k: r[k] for k in SCALARS})
        if (trial + 1) % 10 == 0 or trial + 1 == N_MONTE_CARLO:
            print(f"  Monte Carlo trial {trial+1}/{N_MONTE_CARLO}")
    return out


def run_all_experiments():
    out_dir = os.path.join(RESULTS_ROOT, f'seed_{RANDOM_SEED}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Seed {RANDOM_SEED} -> {out_dir} ===")

    data = generate_all_datasets()
    t0 = time.time(); bb_model, bb_hist = train_blackbox(data); bb_time = time.time() - t0
    print(f"Black-box training time: {bb_time:.1f}s")
    t0 = time.time(); pinn_model, pinn_hist = train_pinn(data); pinn_time = time.time() - t0
    print(f"PINN training time: {pinn_time:.1f}s")

    ol = {'blackbox': compute_open_loop_metrics(bb_model, data),
          'pinn': compute_open_loop_metrics(pinn_model, data)}
    print("\nOpen-loop one-step:", {k: {m: round(v, 5) for m, v in ol[k]['one_step'].items()} for k in ol})

    builders = {'blackbox': lambda: NeuralMPC(bb_model, data['norm_stats'], 'blackbox'),
                'pinn': lambda: NeuralMPC(pinn_model, data['norm_stats'], 'pinn'),
                'first_principles': lambda: FirstPrinciplesMPC()}
    ctrls = {name: builders[name]() for name in CONTROLLERS}

    cl, traj = {}, {}
    for dist in DISTURBANCE_SCENARIOS:
        cl[dist] = {}
        print(f"\n--- step_{dist} ---")
        for name, c in ctrls.items():
            r = run_closed_loop(c, 'step', dist)
            cl[dist][name] = {k: r[k] for k in SCALARS}
            for key in ['states', 'inputs']:
                traj[f'{dist}__{name}__{key}'] = r[key]
            traj[f'{dist}__reference'] = r['reference']; traj['time'] = r['time']
            print(f"  {name:<17} RMSE_T={r['rmse_T']:7.3f} K  offset_T={r['offset_T']:6.3f} K  "
                  f"RMSE_CA={r['rmse_CA']:.4f}  failed={r['n_failed_solves']}  solve={r['solve_ms_mean']:.1f} ms")

    print("\n--- Monte Carlo ---")
    mc = monte_carlo(ctrls)
    for name in mc:
        v = np.array([t['rmse_T'] for t in mc[name]])
        print(f"  {name:<17} RMSE_T = {v.mean():.3f} +/- {v.std():.3f} K")

    summary = {'seed': RANDOM_SEED, 'config': config_snapshot(),
               'training': {'blackbox': {'time_s': bb_time, 'best_epoch': bb_hist['best_epoch']},
                            'pinn': {'time_s': pinn_time, 'best_epoch': pinn_hist['best_epoch']}},
               'open_loop': ol, 'closed_loop': cl,
               'monte_carlo': {n: {'trials': mc[n],
                                   'mean': {k: float(np.mean([t[k] for t in mc[n]])) for k in SCALARS}}
                               for n in mc},
               'histories': {'blackbox': {k: v for k, v in bb_hist.items() if isinstance(v, list)},
                             'pinn': {k: v for k, v in pinn_hist.items() if isinstance(v, list)}}}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, 'trajectories.npz'), **traj)
    np.savez(os.path.join(out_dir, 'norm_stats.npz'), **data['norm_stats'])
    torch.save(bb_model.state_dict(), os.path.join(out_dir, 'blackbox_model.pt'))
    torch.save(pinn_model.state_dict(), os.path.join(out_dir, 'pinn_model.pt'))

    from analysis import generate_seed_plots
    generate_seed_plots(out_dir)
    print(f"\nSaved results to {out_dir}")
    return summary


if __name__ == '__main__':
    run_all_experiments()
