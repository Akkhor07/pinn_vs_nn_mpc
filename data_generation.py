"""
data_generation.py — Generate training, validation, and test datasets.
"""
import numpy as np
from config import (
    RANDOM_SEED, N_TRAIN_TRAJECTORIES, N_VAL_TRAJECTORIES,
    N_TEST_TRAJECTORIES, TRAJ_LENGTH, DT,
    CA_MIN, CA_MAX, T_MIN, T_MAX, TC_MIN, TC_MAX,
    TC_STEP_MIN, TC_STEP_MAX, TC_CHANGE_PROB,
    NOISE_STD_CA, NOISE_STD_T, SS_CA, SS_T, SS_TC
)
from cstr_plant import simulate_cstr_trajectory


def generate_random_input_sequence(length, rng):
    """Generate a piecewise-constant random input sequence."""
    Tc = np.zeros(length)
    Tc[0] = rng.uniform(TC_STEP_MIN, TC_STEP_MAX)
    for k in range(1, length):
        if rng.random() < TC_CHANGE_PROB:
            Tc[k] = rng.uniform(TC_STEP_MIN, TC_STEP_MAX)
        else:
            Tc[k] = Tc[k - 1]
    return Tc


def generate_random_initial_state(rng):
    """Generate a random initial state near the operating region."""
    CA_0 = rng.uniform(0.2, 0.8)
    T_0 = rng.uniform(320.0, 390.0)
    return np.array([CA_0, T_0])


def generate_dataset(n_trajectories, rng, add_noise=False):
    """
    Generate a dataset of input-output transitions.

    Returns:
        X: (N_total, 3) — [CA_k, T_k, Tc_k]
        Y: (N_total, 2) — [CA_{k+1}, T_{k+1}]
    """
    X_list = []
    Y_list = []

    for i in range(n_trajectories):
        x0 = generate_random_initial_state(rng)
        Tc_seq = generate_random_input_sequence(TRAJ_LENGTH, rng)
        states = simulate_cstr_trajectory(x0, Tc_seq)

        for k in range(TRAJ_LENGTH):
            x_k = states[k].copy()
            x_k1 = states[k + 1].copy()

            if add_noise:
                x_k[0] += rng.normal(0, NOISE_STD_CA)
                x_k[1] += rng.normal(0, NOISE_STD_T)
                x_k1[0] += rng.normal(0, NOISE_STD_CA)
                x_k1[1] += rng.normal(0, NOISE_STD_T)

            X_list.append([x_k[0], x_k[1], Tc_seq[k]])
            Y_list.append([x_k1[0], x_k1[1]])

    return np.array(X_list), np.array(Y_list)


def generate_all_datasets():
    """Generate train, validation, and test datasets."""
    rng = np.random.default_rng(RANDOM_SEED)

    print("Generating training data...")
    X_train, Y_train = generate_dataset(N_TRAIN_TRAJECTORIES, rng, add_noise=True)

    print("Generating validation data...")
    X_val, Y_val = generate_dataset(N_VAL_TRAJECTORIES, rng, add_noise=True)

    print("Generating test data (no noise for clean evaluation)...")
    X_test, Y_test = generate_dataset(N_TEST_TRAJECTORIES, rng, add_noise=False)

    # Compute normalization statistics from training data
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)

    norm_stats = {
        'X_mean': X_mean, 'X_std': X_std,
        'Y_mean': Y_mean, 'Y_std': Y_std,
    }

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"X_mean: {X_mean}, X_std: {X_std}")
    print(f"Y_mean: {Y_mean}, Y_std: {Y_std}")

    data = {
        'X_train': X_train, 'Y_train': Y_train,
        'X_val': X_val, 'Y_val': Y_val,
        'X_test': X_test, 'Y_test': Y_test,
        'norm_stats': norm_stats,
    }

    return data


if __name__ == '__main__':
    data = generate_all_datasets()
    np.savez('cstr_data.npz', **{k: v for k, v in data.items() if k != 'norm_stats'},
             **{f'norm_{k}': v for k, v in data['norm_stats'].items()})
    print("Data saved to cstr_data.npz")