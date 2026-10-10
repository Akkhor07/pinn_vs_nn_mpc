"""
aggregate_seeds.py — Seed-level statistics across all <RESULTS_ROOT>/seed_*/summary.json.

For each scenario and metric, PINN vs black-box (paired over seeds):
  mean +/- SD per controller (first-principles reported descriptively),
  mean paired difference (BB - PINN) with 95% t-CI, Cohen's d_z,
  paired t-test p, exact Wilcoxon signed-rank p, Shapiro-Wilk p on the
  differences (normality check for the t-test), seedwise PINN wins,
  Holm-adjusted t-test p across the 7 primary comparisons of that metric.
Note: with n seeds and all differences of one sign, the smallest possible
two-sided exact Wilcoxon p is 2/2^n (0.0156 for n=7).

Usage:  python aggregate_seeds.py [results_root]
"""
import glob
import json
import os
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCENARIOS = ['open_loop', 'none', 'feed_concentration', 'feed_temperature',
             'parameter_mismatch', 'combined', 'monte_carlo']
LABELS = {'open_loop': 'Open-loop (1-step)', 'none': 'Nominal', 'feed_concentration': 'Feed concentration',
          'feed_temperature': 'Feed temperature', 'parameter_mismatch': 'Parameter mismatch',
          'combined': 'Combined', 'monte_carlo': 'Monte Carlo (mean of trials)'}
METRICS = ['rmse_T', 'rmse_CA', 'offset_T', 'control_effort', 'solve_ms_mean', 'n_failed_solves',
           'target_unreachable_frac', 'ignited']
CTRLS = ['blackbox', 'pinn', 'first_principles']


def value(S, scen, ctrl, metric):
    if scen == 'open_loop':
        if ctrl == 'first_principles' or metric not in ('rmse_T', 'rmse_CA'):
            return np.nan
        return S['open_loop'][ctrl]['one_step'][metric]
    if scen == 'monte_carlo':
        if metric in ('target_unreachable_frac', 'ignited'):   # mean over trials (ignited -> fraction)
            tr = S['monte_carlo'].get(ctrl, {}).get('trials', [])
            return float(np.mean([float(t.get(metric, np.nan)) for t in tr])) if tr else np.nan
        return S['monte_carlo'].get(ctrl, {}).get('mean', {}).get(metric, np.nan)
    v = S['closed_loop'][scen].get(ctrl, {}).get(metric, np.nan)
    return float(v) if v is not None else np.nan


def holm(pvals):
    p = np.asarray(pvals, float); m = np.sum(~np.isnan(p)); out = np.full_like(p, np.nan)
    order = [i for i in np.argsort(p) if not np.isnan(p[i])]
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); out[i] = run
    return out


def paired_stats(bb, pinn):
    d = bb - pinn; n = len(d)
    if n < 2 or np.all(np.isnan(d)):
        return {}
    md, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    res = {'n': n, 'mean_diff': md, 'ci_lo': md - half, 'ci_hi': md + half,
           'dz': md / sd if sd > 0 else np.nan, 'wins': int(np.sum(pinn < bb))}
    res['p_t'] = float(stats.ttest_rel(bb, pinn).pvalue) if sd > 0 else np.nan
    try:
        res['p_wilcoxon'] = float(stats.wilcoxon(bb, pinn, method='exact').pvalue)
    except Exception:  # noqa: BLE001
        res['p_wilcoxon'] = np.nan
    res['p_shapiro'] = float(stats.shapiro(d).pvalue) if n >= 3 and sd > 0 else np.nan
    return res


def main(root):
    files = sorted(glob.glob(os.path.join(root, 'seed_*', 'summary.json')))
    if not files:
        sys.exit(f"No summaries found under {root}/seed_*/")
    S = [json.load(open(f)) for f in files]
    seeds = [s['seed'] for s in S]
    out = os.path.join(root, 'aggregate'); os.makedirs(out, exist_ok=True)
    print(f"Aggregating {len(S)} seeds: {seeds}")

    all_rows = {}
    for metric in METRICS:
        rows = []
        for scen in SCENARIOS:
            v = {c: np.array([value(s, scen, c, metric) for s in S], float) for c in CTRLS}
            r = {'scenario': scen, 'metric': metric}
            for c in CTRLS:
                r[f'{c}_mean'] = float(np.nanmean(v[c])) if not np.all(np.isnan(v[c])) else np.nan
                r[f'{c}_sd'] = float(np.nanstd(v[c], ddof=1)) if np.sum(~np.isnan(v[c])) > 1 else np.nan
            r.update(paired_stats(v['blackbox'], v['pinn']))
            r['per_seed'] = {c: v[c].tolist() for c in CTRLS}
            rows.append(r)
        adj = holm([r.get('p_t', np.nan) for r in rows])
        for r, a in zip(rows, adj):
            r['p_t_holm'] = float(a)
        all_rows[metric] = rows

        cols = ['scenario', 'blackbox_mean', 'blackbox_sd', 'pinn_mean', 'pinn_sd',
                'first_principles_mean', 'first_principles_sd', 'mean_diff', 'ci_lo', 'ci_hi',
                'dz', 'wins', 'n', 'p_t', 'p_t_holm', 'p_wilcoxon', 'p_shapiro']
        with open(os.path.join(out, f'stats_{metric}.csv'), 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')

    for metric in ('target_unreachable_frac', 'ignited'):
        print(f"\n{metric} (mean over seeds; Monte Carlo = mean over trials):")
        for r in all_rows[metric]:
            if r['scenario'] == 'open_loop':
                continue
            print(f"  {LABELS[r['scenario']]:<28} BB {r['blackbox_mean']:.3f}  PINN {r['pinn_mean']:.3f}  "
                  f"FP {r['first_principles_mean']:.3f}  PINN lower in {sum(np.array(r['per_seed']['pinn']) < np.array(r['per_seed']['blackbox']))}/{r.get('n', '')} seeds")

    # console summary + LaTeX table for the primary metric
    rows = all_rows['rmse_T']
    print(f"\nRMSE_T (K), mean +/- SD over {len(S)} seeds; diff = BB - PINN (positive favours PINN)")
    hdr = f"{'Scenario':<28}{'BB':>15}{'PINN':>15}{'FP-NMPC':>15}{'diff [95% CI]':>24}{'dz':>7}{'wins':>6}{'p_t':>8}{'p_Holm':>8}{'p_W':>8}"
    print(hdr); print('-' * len(hdr))
    fmt = lambda m, s: '—' if np.isnan(m) else f"{m:.2f}±{s:.2f}"
    for r in rows:
        ci = f"{r.get('mean_diff', np.nan):+.2f} [{r.get('ci_lo', np.nan):+.2f},{r.get('ci_hi', np.nan):+.2f}]"
        print(f"{LABELS[r['scenario']]:<28}{fmt(r['blackbox_mean'], r['blackbox_sd']):>15}{fmt(r['pinn_mean'], r['pinn_sd']):>15}"
              f"{fmt(r['first_principles_mean'], r['first_principles_sd']):>15}{ci:>24}{r.get('dz', np.nan):>7.2f}"
              f"{str(r.get('wins', ''))+'/'+str(r.get('n', '')):>6}{r.get('p_t', np.nan):>8.4f}{r['p_t_holm']:>8.4f}{r.get('p_wilcoxon', np.nan):>8.4f}")

    with open(os.path.join(out, 'table_rmse_T.tex'), 'w') as f:
        f.write("% Auto-generated by aggregate_seeds.py\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllllllll@{}}\n\\toprule\n")
        f.write("Scenario & Black-box & PINN & FP-NMPC & $\\Delta$ (BB$-$PINN) [95\\% CI] & $d_z$ & Wins & $p_t$ (Holm) & $p_W$ \\\\\n\\midrule\n")
        for r in rows:
            g = lambda m, s: '---' if np.isnan(m) else f"${m:.2f}\\pm{s:.2f}$"
            f.write(f"{LABELS[r['scenario']]} & {g(r['blackbox_mean'], r['blackbox_sd'])} & {g(r['pinn_mean'], r['pinn_sd'])} & "
                    f"{g(r['first_principles_mean'], r['first_principles_sd'])} & "
                    f"${r['mean_diff']:+.2f}$ [${r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}$] & {r['dz']:.2f} & "
                    f"{r['wins']}/{r['n']} & {r['p_t']:.4f} ({r['p_t_holm']:.4f}) & {r['p_wilcoxon']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular*}\n")

    # seed-variance figure (one panel per primary metric)
    colors = {'blackbox': '#1f77b4', 'pinn': '#d62728', 'first_principles': '0.4'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, metric, ylab in [(axes[0], 'rmse_T', r'RMSE$_T$ (K)'), (axes[1], 'offset_T', 'Steady-state offset T (K)')]:
        for i, r in enumerate(all_rows[metric]):
            for j, c in enumerate(CTRLS):
                vals = np.array(r['per_seed'][c], float); vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                x = i + (j - 1) * 0.27
                bp = ax.boxplot(vals, positions=[x], widths=0.22, patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(colors[c]); bp['boxes'][0].set_alpha(0.35)
                ax.scatter(x + np.linspace(-0.05, 0.05, len(vals)), vals, s=12, color=colors[c], zorder=3)
        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels([LABELS[s].replace(' (', '\n(').replace(' ', '\n', 1) for s in SCENARIOS], fontsize=8)
        ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis='y'); ax.set_ylim(bottom=0)
    for c, lab in [('blackbox', 'Black-box NN'), ('pinn', 'PINN'), ('first_principles', 'First-principles NMPC')]:
        axes[0].scatter([], [], color=colors[c], label=lab)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig5_seed_variance.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    with open(os.path.join(out, 'all_stats.json'), 'w') as f:
        json.dump({'seeds': seeds, 'stats': all_rows}, f, indent=1, default=float)
    print(f"\nWrote CSVs, LaTeX table, figure and all_stats.json to {out}")


if __name__ == '__main__':
    from config import RESULTS_ROOT
    main(sys.argv[1] if len(sys.argv) > 1 else RESULTS_ROOT)