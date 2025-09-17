import matplotlib.pyplot as plt
import numpy as np
from src.linear_regression import LinearRegressionGD, StandardScaler
from src.utils import read_csv_xy, train_test_split

# ===============================
# Configuración
# ===============================
CSV_PATH = "data/toy_data_complex.csv"
TARGET = "price"
ALPHA = 0.01
ITERS = 4000
SEED = 42

# ===============================
# Cargar datos
# ===============================
X, y, feature_names = read_csv_xy(CSV_PATH, TARGET)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, seed=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, seed=SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ===============================
# Entrenar modelos
# ===============================
model_plain = LinearRegressionGD(alpha=ALPHA, iters=ITERS, fit_intercept=True, l2=0.0)
model_plain.fit(X_train, y_train)

model_ridge = LinearRegressionGD(alpha=ALPHA, iters=ITERS, fit_intercept=True, l2=5.0)
model_ridge.fit(X_train, y_train)

# ===============================
# Evaluación
# ===============================
def resumen(model, X_tr, y_tr, X_val, y_val, X_te, y_te, label):
    def eval_set(X, y):
        yhat = model.predict(X)
        mse = model.mse(y, yhat)
        mae = model.mae(y, yhat)
        r2  = model.r2(y, yhat)
        return mse, mae, r2
    tr = eval_set(X_tr, y_tr)
    va = eval_set(X_val, y_val)
    te = eval_set(X_te, y_te)
    print(f"\n=== {label} ===")
    print(f"Train -> MSE {tr[0]:.2f} | MAE {tr[1]:.2f} | R² {tr[2]:.3f}")
    print(f"Val   -> MSE {va[0]:.2f} | MAE {va[1]:.2f} | R² {va[2]:.3f}")
    print(f"Test  -> MSE {te[0]:.2f} | MAE {te[1]:.2f} | R² {te[2]:.3f}")
    return tr, va, te

plain_results = resumen(model_plain, X_train, y_train, X_val, y_val, X_test, y_test, "Modelo sin regularización")
ridge_results = resumen(model_ridge, X_train, y_train, X_val, y_val, X_test, y_test, "Modelo con Ridge")

# ===============================
# Gráficas comparativas
# ===============================
labels = ["Train", "Validation", "Test"]

def plot_comparison(metric_idx, metric_name):
    plain_vals = [plain_results[i][metric_idx] for i in range(3)]
    ridge_vals = [ridge_results[i][metric_idx] for i in range(3)]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, plain_vals, width, label="Normal")
    ax.bar(x + width/2, ridge_vals, width, label="Ridge")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(metric_name)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"comp_{metric_name}.png")
    plt.close()

plot_comparison(0, "MSE")
plot_comparison(1, "MAE")
plot_comparison(2, "R2")

print("\nGráficas guardadas: comp_MSE.png, comp_MAE.png, comp_R2.png")
