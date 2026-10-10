"""
run_seeds.py — Run the full pipeline for several seeds, each in a FRESH Python
process (so RANDOM_SEED is re-read), then aggregate.

    python run_seeds.py                      # default 7 seeds
    python run_seeds.py 42 45 50             # chosen seeds
    python run_seeds.py --sweep [seeds...]   # also run the disturbance-magnitude sweep per seed
Set PINN_SMOKE=1 for a fast code check.
"""
import os
import subprocess
import sys

SEEDS = [42, 45, 50, 75, 100, 28, 343]

if __name__ == '__main__':
    sweep = '--sweep' in sys.argv
    seeds = [int(s) for s in sys.argv[1:] if s != '--sweep'] or SEEDS
    from config import RESULTS_ROOT
    for s in seeds:
        log_dir = os.path.join(RESULTS_ROOT, f'seed_{s}'); os.makedirs(log_dir, exist_ok=True)
        env = dict(os.environ, PINN_SEED=str(s))
        print(f"\n######## seed {s} ########", flush=True)
        with open(os.path.join(log_dir, 'log.txt'), 'w') as log:
            p = subprocess.Popen([sys.executable, '-u', 'run_experiments.py'], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                print(line, end=''); log.write(line)
            if p.wait() != 0:
                sys.exit(f"seed {s} failed; see {log_dir}/log.txt")
        if sweep:
            subprocess.run([sys.executable, '-u', 'sweep_disturbances.py', log_dir], env=env, check=True)
    subprocess.run([sys.executable, 'aggregate_seeds.py', RESULTS_ROOT], check=True)
    if sweep:
        subprocess.run([sys.executable, 'sweep_disturbances.py', '--aggregate', RESULTS_ROOT], check=True)
