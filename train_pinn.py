"""
train_pinn.py — PINN training: data loss + lambda(epoch) * physics residual.
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import (N_EPOCHS_PINN, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
                    LAMBDA_PHYSICS, LAMBDA_WARMUP_FRAC, LAMBDA_MIN_FRAC,
                    N_COLLOCATION, N_COLLOCATION_POOL, COLLOCATION_RANGES,
                    PHYSICS_TARGET_MODE, RANDOM_SEED, physics_prior_params)
from models import PhysicsInformedNN, precompute_physics_targets
from cstr_plant import simulate_cstr_step


def get_lambda_physics(epoch, total_epochs, lambda_max=LAMBDA_PHYSICS,
                       lambda_min_frac=LAMBDA_MIN_FRAC, warmup_frac=LAMBDA_WARMUP_FRAC):
    """Hold lambda_max for warmup_frac of epochs, then decay linearly to lambda_max*lambda_min_frac."""
    warm = int(total_epochs * warmup_frac)
    if epoch < warm:
        return lambda_max
    prog = (epoch - warm) / max(1, total_epochs - warm)
    return lambda_max - (lambda_max - lambda_max * lambda_min_frac) * prog


def sample_collocation(n, rng):
    r = COLLOCATION_RANGES
    return np.stack([rng.uniform(*r['CA'], n), rng.uniform(*r['T'], n),
                     rng.uniform(*r['Tc'], n)], axis=1)


def validate_targets(X, Y, rng, n_check=100):
    """Spot-check precomputed targets against scipy RK45 (physics-prior params)."""
    idx = rng.choice(len(X), size=min(n_check, len(X)), replace=False)
    p = physics_prior_params()
    ref = np.array([simulate_cstr_step(X[i, :2], X[i, 2], params=p) for i in idx])
    err = np.abs(Y[idx] - ref)
    return float(err[:, 0].max()), float(err[:, 1].max())


def train_pinn(data):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    norm = data['norm_stats']
    nz = lambda X: (X - norm['X_mean']) / norm['X_std']

    Xtr = torch.FloatTensor(nz(data['X_train']))
    Ytr = torch.FloatTensor((data['Y_train'] - norm['Y_mean']) / norm['Y_std'])
    Xva = torch.FloatTensor(nz(data['X_val']))
    Yva = torch.FloatTensor((data['Y_val'] - norm['Y_mean']) / norm['Y_std'])

    precomputed = PHYSICS_TARGET_MODE == 'precomputed'
    if precomputed:
        t0 = time.time()
        Phi_tr = precompute_physics_targets(data['X_train'])
        X_pool = sample_collocation(N_COLLOCATION_POOL, rng)
        Phi_pool = precompute_physics_targets(X_pool)
        eCA, eT = validate_targets(X_pool, Phi_pool, rng)
        print(f"Physics targets precomputed for {len(Phi_tr)} train + {len(X_pool)} collocation "
              f"points in {time.time()-t0:.1f}s; check vs scipy RK45: max|eCA|={eCA:.2e}, max|eT|={eT:.3f} K")
        Phi_tr_t = torch.FloatTensor(Phi_tr)
        Xpool_t = torch.FloatTensor(nz(X_pool))
        Phipool_t = torch.FloatTensor(Phi_pool)
        idx_tr = torch.arange(len(Xtr))
        loader = DataLoader(TensorDataset(Xtr, Ytr, idx_tr), batch_size=BATCH_SIZE, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)

    model = PhysicsInformedNN(norm_stats=norm)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad],
                     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-6)
    mse = nn.MSELoss()

    hist = {'train_losses': [], 'val_losses': [], 'physics_losses': [], 'lambdas': []}
    best_val, best_state, best_epoch = float('inf'), None, -1
    for epoch in range(N_EPOCHS_PINN):
        lam = get_lambda_physics(epoch, N_EPOCHS_PINN)
        model.train()
        dsum = psum = 0.0; nb = 0
        for batch in loader:
            opt.zero_grad()
            if precomputed:
                xb, yb, ib = batch
                ci = torch.as_tensor(rng.integers(0, len(Xpool_t), N_COLLOCATION))
                phys = model.physics_loss_precomputed(xb, Phi_tr_t[ib]) + \
                    model.physics_loss_precomputed(Xpool_t[ci], Phipool_t[ci])
            else:
                xb, yb = batch
                xc = torch.FloatTensor(nz(sample_collocation(N_COLLOCATION, rng)))
                phys = model.physics_loss_online(xb) + model.physics_loss_online(xc)
            dl = mse(model(xb), yb)
            (dl + lam * phys).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            dsum += dl.item(); psum += phys.item(); nb += 1
        model.eval()
        with torch.no_grad():
            v = mse(model(Xva), Yva).item()
        hist['train_losses'].append(dsum / nb)
        hist['physics_losses'].append(psum / nb)
        hist['val_losses'].append(v)
        hist['lambdas'].append(lam)
        sched.step(v)
        if v < best_val:   # selection on pure data validation loss (independent of lambda)
            best_val, best_epoch = v, epoch
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 50 == 0:
            print(f"PINN Epoch {epoch+1}/{N_EPOCHS_PINN} - Data: {dsum/nb:.6f}, "
                  f"Phys: {psum/nb:.6f}, Val: {v:.6f}, Lambda: {lam:.4f}")

    model.load_state_dict(best_state)
    print(f"PINN best checkpoint: epoch {best_epoch+1} (val={best_val:.6f})")
    hist.update({'best_epoch': best_epoch + 1, 'best_val': best_val})
    return model, hist
