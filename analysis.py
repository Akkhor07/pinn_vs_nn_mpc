"""
analysis.py — Per-seed figures, generated from saved results only:
    python analysis.py results/seed_45
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.labelsize': 11,
                            'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
STYLE = {'blackbox': dict(color='#1f77b4', label='Black-box NN MPC'),
         'pinn': dict(color='#d62728', label='PINN MPC'),
         'first_principles': dict(color='0.25', ls='--', label='First-principles NMPC')}


def _save(fig, out_dir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'))
    plt.close(fig)


def plot_training_curves(S, out_dir):
    hb, hp = S['histories']['blackbox'], S['histories']['pinn']
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    ax[0].semilogy(hb['train_losses'], color=STYLE['blackbox']['color'], label='Black-box')
    ax[0].semilogy(hp['train_losses'], color=STYLE['pinn']['color'], label='PINN (data term)')
    ax[0].set_title('(a) Training data loss')
    ax[1].semilogy(hb['val_losses'], color=STYLE['blackbox']['color'], label='Black-box')
    ax[1].semilogy(hp['val_losses'], color=STYLE['pinn']['color'], label='PINN')
    for name, h in [('blackbox', hb), ('pinn', hp)]:
        e = S['training'][name]['best_epoch'] - 1
        ax[1].plot(e, h['val_losses'][e], 'o', color=STYLE[name]['color'], ms=5)
    ax[1].set_title('(b) Validation loss (o = selected epoch)')
    ax[2].semilogy(hp['physics_losses'], color='#2ca02c'); ax[2].set_title('(c) PINN physics residual')
    ax[3].plot(hp['lambdas'], color='k'); ax[3].set_title(r'(d) Physics weight $\lambda$')
    for a in ax:
        a.set_xlabel('Epoch'); a.grid(alpha=0.3)
    ax[0].legend(); ax[1].legend()
    fig.tight_layout(); _save(fig, out_dir, 'fig2_training_curves')


def plot_open_loop(S, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (var, unit) in enumerate([('CA', 'mol/L'), ('T', 'K')]):
        for name in ['blackbox', 'pinn']:
            r = S['open_loop'][name]['rollout']
            m, s = np.array(r[f'mean_{var}']), np.array(r[f'std_{var}'])
            k = np.arange(1, len(m) + 1)
            ax[j].plot(k, m, 'o-', ms=3, color=STYLE[name]['color'], label=STYLE[name]['label'].replace(' MPC', ''))
            # absolute error cannot be negative: clip the lower band at 0
            ax[j].fill_between(k, np.maximum(m - s, 0), m + s, color=STYLE[name]['color'], alpha=0.15)
        ax[j].set_xlabel('Prediction step (each 0.5 min)')
        ax[j].set_ylabel(f'Absolute error ({unit})')
        ax[j].set_title(f'({"ab"[j]}) Free-run rollout error, {var} (mean ± 1 SD, 50 trajectories)')
        ax[j].set_ylim(bottom=0); ax[j].grid(alpha=0.3); ax[j].legend()
    fig.tight_layout(); _save(fig, out_dir, 'fig3_open_loop_rollout')


def plot_closed_loop(T, S, dist, out_dir):
    t = T['time']; ref = T[f'{dist}__reference']
    names = [n for n in STYLE if f'{dist}__{n}__states' in T]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t[1:], ref[:, 0], 'k:', lw=1.5, label='Reference')
    ax[1].plot(t[1:], ref[:, 1], 'k:', lw=1.5, label='Reference')
    for n in names:
        x, u = T[f'{dist}__{n}__states'], T[f'{dist}__{n}__inputs']
        rm = S['closed_loop'][dist][n]['rmse_T']
        st = dict(STYLE[n]); st['label'] = f"{st['label']} (RMSE$_T$={rm:.2f} K)"
        ax[0].plot(t, x[:, 0], lw=1.3, **{k: v for k, v in STYLE[n].items()})
        ax[1].plot(t, x[:, 1], lw=1.3, **st)
        ax[2].step(t[:-1], u, where='post', lw=1.1, **{k: v for k, v in STYLE[n].items() if k != 'label'})
    ax[0].set_ylabel(r'$C_A$ (mol/L)'); ax[1].set_ylabel('T (K)'); ax[2].set_ylabel(r'$T_c$ (K)')
    ax[2].set_xlabel('Time (min)')
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(loc='upper right', ncol=2); ax[1].legend(loc='upper right', fontsize=8)
    fig.tight_layout(); _save(fig, out_dir, f'fig4_closed_loop_{dist}')


def plot_mc(S, out_dir):
    names = list(S['monte_carlo'].keys())
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for j, (k, lab) in enumerate([('rmse_T', 'RMSE T (K)'), ('offset_T', 'Steady-state offset T (K)'),
                                  ('control_effort', 'Control effort (K)')]):
        vals = [[t[k] for t in S['monte_carlo'][n]['trials']] for n in names]
        bp = ax[j].boxplot(vals, patch_artist=True)
        ax[j].set_xticks(range(1, len(names) + 1)); ax[j].set_xticklabels([STYLE[n]['label'].replace(' MPC', '').replace(' NMPC', '') for n in names], fontsize=8)
        for b, n in zip(bp['boxes'], names):
            b.set_facecolor(STYLE[n]['color']); b.set_alpha(0.5)
        ax[j].set_ylabel(lab); ax[j].grid(alpha=0.3, axis='y')
    fig.tight_layout(); _save(fig, out_dir, 'fig_mc_seed')


def generate_seed_plots(out_dir):
    with open(os.path.join(out_dir, 'summary.json')) as f:
        S = json.load(f)
    T = np.load(os.path.join(out_dir, 'trajectories.npz'))
    plot_training_curves(S, out_dir)
    plot_open_loop(S, out_dir)
    for dist in S['closed_loop']:
        plot_closed_loop(T, S, dist, out_dir)
    plot_mc(S, out_dir)
    print(f"Figures written to {out_dir}")


if __name__ == '__main__':
    generate_seed_plots(sys.argv[1] if len(sys.argv) > 1 else 'results/seed_45')
