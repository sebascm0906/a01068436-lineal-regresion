import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train(args):
    # Leer dataset
    df = pd.read_csv(args.csv)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Escalado opcional
    scaler = None
    if not args.no_normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Modelo: Linear o Ridge
    if args.ridge > 0.0:
        model = Ridge(alpha=args.ridge, fit_intercept=not args.no_intercept)
    else:
        model = LinearRegression(fit_intercept=not args.no_intercept)

    model.fit(X_train, y_train)

    # Predicciones
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    # Métricas
    def resumen(y_true, y_pred, label):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{label} -> MSE={mse:.2f} | MAE={mae:.2f} | R²={r2:.3f}")
        return {"MSE": mse, "MAE": mae, "R2": r2}

    res_train = resumen(y_train, yhat_train, "Train")
    res_test = resumen(y_test, yhat_test, "Test")

    # Guardar predicciones
    pred_path = "sklearn_predicciones_test.csv"
    pd.DataFrame({"y_true": y_test, "y_pred": yhat_test}).to_csv(pred_path, index=False)
    print(f"Predicciones guardadas en {pred_path}")

    # Guardar modelo y métricas en JSON
    payload = {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "ridge": args.ridge,
        "config": {
            "csv": args.csv,
            "target": args.target,
            "test_size": args.test_size,
            "seed": args.seed,
            "normalized": not args.no_normalize,
            "fit_intercept": not args.no_intercept,
        },
        "results": {
            "train": res_train,
            "test": res_test,
        }
    }
    if scaler is not None:
        payload["scaler"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    with open("sklearn_model.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Modelo y métricas guardados en 'sklearn_model.json'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regresión Lineal con sklearn")
    parser.add_argument("--csv", required=True, help="Ruta al CSV con datos")
    parser.add_argument("--target", required=True, help="Columna objetivo")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción para test")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--no-normalize", action="store_true", help="Desactiva normalización")
    parser.add_argument("--no-intercept", action="store_true", help="Desactiva intercepto")
    parser.add_argument("--ridge", type=float, default=0.0, help="Coeficiente L2 (Ridge)")
    args = parser.parse_args()
    train(args)
