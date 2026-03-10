import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_DIR = "data/train"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

all_data = []

for file in os.listdir(TRAIN_DIR):
    if file.endswith(".csv"):
        path = os.path.join(TRAIN_DIR, file)
        df = pd.read_csv(path)
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

X = data[["day", "temperature"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X_scaled, columns=["day", "temperature"])

joblib.dump(scaler, "scaler.pkl")

scaled_df.to_csv(os.path.join(PROCESSED_DIR, "train_scaled.csv"), index=False)