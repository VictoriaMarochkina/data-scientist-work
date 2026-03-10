import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/processed/train_scaled.csv")

X = data[["day"]]
y = data["temperature"]

model = LinearRegression()
model.fit(X, y)

# сохранение модели
joblib.dump(model, "model.pkl")
