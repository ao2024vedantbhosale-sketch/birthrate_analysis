import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("india_birth_rate.csv").dropna().sort_values("year")
X, y = df[["year"]].values, df["birth_rate"].values

s = int(len(df) * 0.8)
X_train, X_test, y_train, y_test = X[:s], X[s:], y[:s], y[s:]

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Polynomial (degree 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}

best_name, best_mae = None, float("inf")
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    mae, r2 = mean_absolute_error(y_test, p), r2_score(y_test, p)
    print(f"{name}: MAE={mae:.3f} R2={r2:.3f}")
    if mae < best_mae:
        best_name, best_mae = name, mae

print("Best model:", best_name)
best = models[best_name]
best.fit(X, y)
os.makedirs("model", exist_ok=True)
joblib.dump(best, "model/birth_model.joblib")
print("Model saved")
