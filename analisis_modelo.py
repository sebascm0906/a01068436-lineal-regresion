import csv
import random
import matplotlib.pyplot as plt
from src.linear_regression import LinearRegressionGD, StandardScaler
from src.utils import read_csv_xy, train_test_split

# ===============================
# Configuración
# ===============================
CSV_PATH = "data/toy_data_complex.csv"   # dataset
TARGET = "price"
ALPHA = 0.01
ITERS = 4000
SEED = 42

# ===============================
# Cargar datos
# ===============================
X, y, feature_names = read_csv_xy(CSV_PATH, TARGET)

# Train / Validation / Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, seed=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, seed=SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ===============================
# Entrenamiento
# ===============================
model = LinearRegressionGD(alpha=ALPHA, iters=ITERS, fit_intercept=True)
model.fit(X_train, y_train)

# Predicciones
yhat_train = model.predict(X_train)
yhat_val   = model.predict(X_val)
yhat_test  = model.predict(X_test)

# ===============================
# Métricas
# ===============================
def resumen(y_true, y_pred, label):
    mse = model.mse(y_true, y_pred)
    mae = model.mae(y_true, y_pred)
    r2  = model.r2(y_true, y_pred)
    print(f"{label} -> MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
    return {"MSE": mse, "MAE": mae, "R2": r2}

train_res = resumen(y_train, yhat_train, "Train")
val_res   = resumen(y_val,   yhat_val,   "Validation")
test_res  = resumen(y_test,  yhat_test,  "Test")

# ===============================
# Gráficas
# ===============================

# 1) Reales vs Predichos
plt.figure(figsize=(6,6))
plt.scatter(y_test, yhat_test, alpha=0.7, color="steelblue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Valores reales vs predichos (Test)")
plt.tight_layout()
plt.savefig("graf_reales_vs_pred.png")
plt.close()

# 2) Distribución de residuos
residuos = [yt - yp for yt, yp in zip(y_test, yhat_test)]
plt.figure(figsize=(6,4))
plt.hist(residuos, bins=30, color="gray", edgecolor="black")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residuo (y_real - y_predicho)")
plt.ylabel("Frecuencia")
plt.title("Distribución de residuos (Test)")
plt.tight_layout()
plt.savefig("graf_residuos.png")
plt.close()

# 3) Comparación de métricas
import numpy as np
labels = ["Train", "Validation", "Test"]
mse_vals = [train_res["MSE"], val_res["MSE"], test_res["MSE"]]
mae_vals = [train_res["MAE"], val_res["MAE"], test_res["MAE"]]
r2_vals  = [train_res["R2"],  val_res["R2"],  test_res["R2"]]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(1,3, figsize=(15,4))

ax[0].bar(x, mse_vals, width, color="skyblue")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_title("MSE")

ax[1].bar(x, mae_vals, width, color="lightgreen")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_title("MAE")

ax[2].bar(x, r2_vals, width, color="salmon")
ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].set_title("R²")

plt.suptitle("Comparación de métricas por conjunto")
plt.tight_layout()
plt.savefig("graf_metricas.png")
plt.close()

print("\nGráficas guardadas: graf_reales_vs_pred.png, graf_residuos.png, graf_metricas.png")
