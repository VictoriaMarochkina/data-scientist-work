import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

TEST_DIR = "data/test"

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

all_predictions = []
all_true = []

for file in os.listdir(TEST_DIR):
    if file.endswith(".csv"):
        path = os.path.join(TEST_DIR, file)

        df = pd.read_csv(path)

        X = df[["day", "temperature"]]

        # масштабируем данные
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=["day", "temperature"])

        X_test = X_scaled_df[["day"]]
        y_test = X_scaled_df["temperature"]

        predictions = model.predict(X_test)

        all_predictions.extend(predictions)
        all_true.extend(y_test)

# считаем метрики
mse = mean_squared_error(all_true, all_predictions)
r2 = r2_score(all_true, all_predictions)

print("MSE:", mse)
print("R2:", r2)

