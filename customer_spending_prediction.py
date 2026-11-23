import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar el dataset
df = pd.read_csv("customer_spending.csv")

# 2. Revisar dimensiones, tipos, nulos y duplicados
print("Dimensiones (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores nulos por columna:\n", df.isnull().sum())
print("\nFilas duplicadas:", df.duplicated().sum())

# 3. Estadísticas descriptivas y correlación con annual_spending
print("\nResumen estadístico:\n", df.describe().round(2))
print("\nCorrelación con annual_spending:\n", df.corr(numeric_only=True)["annual_spending"].sort_values(ascending=False).round(4))

# 4. Separar predictoras (X) y variable objetivo (y)
features = ["age", "monthly_visits", "avg_order_value", "months_customer", "emails_opened"]
X = df[features]
y = df["annual_spending"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. División entrenamiento / prueba (80% / 20%) antes de ajustar modelos (sin data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Definir los tres modelos solicitados
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

# 3. Entrenar y evaluar cada modelo en el mismo conjunto de prueba
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    })

metrics_df = pd.DataFrame(results)
print("Tabla comparativa en Test:\n")
print(metrics_df.to_string(index=False))

# 4. Predicción para el nuevo cliente
new_customer = pd.DataFrame([{
    "age": 40,
    "monthly_visits": 12,
    "avg_order_value": 80,
    "months_customer": 36,
    "emails_opened": 10,
}])

print("\nPredicción de gasto anual para el nuevo cliente:")
for name, model in models.items():
    pred_val = model.predict(new_customer)[0]
    print(f"- {name}: {pred_val:,.2f}")

import matplotlib.pyplot as plt
import numpy as np

# 1. Recolectar predicciones en test y del nuevo cliente
preds_test = {}
new_preds = {}
for name, model in models.items():
    preds_test[name] = model.predict(X_test)
    new_preds[name] = model.predict(new_customer)[0]

colors = ["#2b6cb0", "#dd6b20", "#2f855a"]

# Gráfico 1: Comparación de MAE, RMSE y R² en Test
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

x_pos = np.arange(len(metrics_df["Model"]))
width = 0.35
bars_mae = axes[0].bar(x_pos - width/2, metrics_df["MAE"], width, label="MAE", color="#3182ce")
bars_rmse = axes[0].bar(x_pos + width/2, metrics_df["RMSE"], width, label="RMSE", color="#63b3ed")
axes[0].set_title("Error Metrics on Test Set (Lower is Better)")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(metrics_df["Model"])
axes[0].set_ylabel("Annual Spending Units")
axes[0].legend()
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
for b in bars_mae:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 4, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for b in bars_rmse:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 4, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)

bars_r2 = axes[1].bar(metrics_df["Model"], metrics_df["R2"], width=0.45, color=colors)
axes[1].set_title("R² Score on Test Set (Higher is Better)")
axes[1].set_ylim(0.90, 1.01)
axes[1].set_ylabel("R² Score")
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
for b in bars_r2:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.002, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# Gráfico 2: Valores reales vs. predichos en Test y predicción del nuevo cliente
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.8))

idx = np.arange(len(y_test))
axes2[0].plot(idx, y_test.values, marker="o", label="Actual (Test)", color="#1a202c", linewidth=2)
axes2[0].plot(idx, preds_test["Linear Regression"], marker="s", linestyle="--", label="Linear Regression", color="#2b6cb0")
axes2[0].plot(idx, preds_test["Random Forest"], marker="^", linestyle=":", label="Random Forest", color="#2f855a")
axes2[0].plot(idx, preds_test["Decision Tree"], marker="x", linestyle="-.", label="Decision Tree", color="#dd6b20")
axes2[0].set_title("Actual vs. Predicted Values (Test Set)")
axes2[0].set_xlabel("Test Sample Index")
axes2[0].set_ylabel("Annual Spending")
axes2[0].set_xticks(idx)
axes2[0].legend(fontsize=8)
axes2[0].grid(True, linestyle="--", alpha=0.4)

model_names = list(new_preds.keys())
pred_vals = list(new_preds.values())
bars_nc = axes2[1].bar(model_names, pred_vals, width=0.45, color=colors)
axes2[1].set_title("New Customer Predicted Annual Spending")
axes2[1].set_ylabel("Predicted Annual Spending")
axes2[1].grid(axis="y", linestyle="--", alpha=0.4)
for b in bars_nc:
    axes2[1].text(b.get_x() + b.get_width()/2, b.get_height() + 40, f"{b.get_height():,.2f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
plt.savefig("predictions_overview.png", dpi=150)
plt.show()