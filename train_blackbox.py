"""
train_blackbox.py — Training loop for the black-box network.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import N_EPOCHS_BLACKBOX, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, RANDOM_SEED
from models import BlackBoxNN


def train_blackbox(data):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    norm = data['norm_stats']
    Xtr = torch.FloatTensor((data['X_train'] - norm['X_mean']) / norm['X_std'])
    Ytr = torch.FloatTensor((data['Y_train'] - norm['Y_mean']) / norm['Y_std'])
    Xva = torch.FloatTensor((data['X_val'] - norm['X_mean']) / norm['X_std'])
    Yva = torch.FloatTensor((data['Y_val'] - norm['Y_mean']) / norm['Y_std'])
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)

    model = BlackBoxNN()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, min_lr=1e-6)
    mse = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val, best_state, best_epoch = float('inf'), None, -1
    for epoch in range(N_EPOCHS_BLACKBOX):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = mse(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        train_losses.append(tot / nb)
        model.eval()
        with torch.no_grad():
            v = mse(model(Xva), Yva).item()
        val_losses.append(v)
        sched.step(v)
        if v < best_val:
            # BUGFIX: state_dict().copy() was a SHALLOW copy -> tensors kept
            # being updated in place, so the "best" checkpoint silently became
            # the final-epoch weights. deepcopy freezes the true best weights.
            best_val, best_epoch = v, epoch
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 50 == 0:
            print(f"BlackBox Epoch {epoch+1}/{N_EPOCHS_BLACKBOX} - Train: {train_losses[-1]:.6f}, Val: {v:.6f}")

    model.load_state_dict(best_state)
    print(f"Black-box best checkpoint: epoch {best_epoch+1} (val={best_val:.6f})")
    return model, {'train_losses': train_losses, 'val_losses': val_losses,
                   'best_epoch': best_epoch + 1, 'best_val': best_val}
