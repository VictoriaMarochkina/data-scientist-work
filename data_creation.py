import os
import numpy as np
import pandas as pd

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def generate_dataset(n_days=365, noise_level=1.5, add_anomalies=False):
    days = np.arange(n_days)

    temperature = 10 + 10 * np.sin(2 * np.pi * days / 365)

    # Шум
    noise = np.random.normal(0, noise_level, n_days)
    temperature = temperature + noise

    # Аномалии
    if add_anomalies:
        n_anomalies = np.random.randint(3, 8)
        anomaly_indices = np.random.choice(n_days, n_anomalies, replace=False)
        temperature[anomaly_indices] += np.random.choice([15, -15], size=n_anomalies)

    df = pd.DataFrame({
        "day": days,
        "temperature": temperature
    })

    return df


N_DATASETS = 20

for i in range(N_DATASETS):

    anomalies = np.random.rand() < 0.4
    df = generate_dataset(add_anomalies=anomalies)

    if i < int(0.7 * N_DATASETS):
        path = f"{TRAIN_DIR}/dataset_{i}.csv"
    else:
        path = f"{TEST_DIR}/dataset_{i}.csv"

    df.to_csv(path, index=False)