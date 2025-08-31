import argparse
import json
import os
from typing import List

from utils import read_csv_xy, train_test_split, write_predictions_csv
from linear_regression import LinearRegressionGD, StandardScaler, save_model, load_model

def train_cmd(args):
    X, y, feature_names = read_csv_xy(args.csv, args.target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, seed=args.seed)

    scaler = None if args.no_normalize else StandardScaler()
    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    model = LinearRegressionGD(alpha=args.alpha, iters=args.iters, fit_intercept=not args.no_intercept)
    model.fit(X_tr, y_tr)

    # Métricas
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)
    mse_tr = model.mse(y_tr, yhat_tr)
    mae_tr = model.mae(y_tr, yhat_tr)
    r2_tr = model.r2(y_tr, yhat_tr)

    mse_te = model.mse(y_te, yhat_te)
    mae_te = model.mae(y_te, yhat_te)
    r2_te = model.r2(y_te, yhat_te)

    print("=== Entrenamiento ===")
    print(f"MSE: {mse_tr:.4f} | MAE: {mae_tr:.4f} | R2: {r2_tr:.4f}")
    print("=== Prueba ===")
    print(f"MSE: {mse_te:.4f} | MAE: {mae_te:.4f} | R2: {r2_te:.4f}")
    print()

    # Muestra algunas predicciones vs reales
    k = min(5, len(y_te))
    print("Ejemplos de predicciones (test):")
    for i in range(k):
        print(f"y_true={y_te[i]:.3f} | y_pred={yhat_te[i]:.3f}")

    # Guardar predicciones de test
    pred_path = "predicciones_test.csv"
    write_predictions_csv(pred_path, y_te, yhat_te)
    print(f"\nSe guardó {pred_path} en el directorio actual.")

    # Guardar modelo
    config = {
        "csv": args.csv,
        "target": args.target,
        "alpha": args.alpha,
        "iters": args.iters,
        "test_size": args.test_size,
        "normalized": not args.no_normalize,
        "fit_intercept": not args.no_intercept,
        "seed": args.seed,
    }
    save_model("model.json", model, scaler, feature_names, config)
    print("Modelo guardado en 'model.json'.")

def parse_features(s: str) -> List[float]:
    parts = [p for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]

def predict_cmd(args):
    if not os.path.exists(args.model):
        raise SystemExit(f"No se encontró el archivo de modelo: {args.model}")
    model, scaler, feature_names = load_model(args.model)
    if args.csv:
        # CSV de nuevas muestras con encabezados == feature_names
        import csv
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if feature_names is None:
                feature_names = reader.fieldnames
            if reader.fieldnames != feature_names:
                raise SystemExit(f"Las columnas del CSV no coinciden con las usadas en entrenamiento.\nEsperado: {feature_names}\nRecibido: {reader.fieldnames}")
            X_new = []
            for row in reader:
                X_new.append([float(row[c]) for c in feature_names])
        if scaler is not None:
            X_new = scaler.transform(X_new)
        y_pred = model.predict(X_new)
        out = args.out or "predicciones_nuevas.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y_pred"])
            for p in y_pred:
                w.writerow([p])
        print(f"Predicciones guardadas en '{out}'. Filas: {len(y_pred)}")
        if args.take_first:
            print("Primeras filas:")
            for i in range(min(args.take_first, len(y_pred))):
                print(f"{i}: {y_pred[i]:.4f}")
    elif args.features:
        x = parse_features(args.features)
        if scaler is not None:
            # Necesitamos el mismo número de características que en entrenamiento
            if feature_names is not None and len(x) != len(feature_names):
                raise SystemExit(f"Se esperaban {len(feature_names)} características, recibidas {len(x)}.")
            Xn = scaler.transform([x])
        else:
            Xn = [x]
        y_pred = model.predict(Xn)[0]
        print(f"Predicción: {y_pred:.6f}")
    else:
        raise SystemExit("Usa --csv o --features.")

def build_parser():
    p = argparse.ArgumentParser(description="Regresión Lineal desde cero (GD)")
    sp = p.add_subparsers(dest="cmd", required=True)

    pt = sp.add_parser("train", help="Entrenar y evaluar el modelo")
    pt.add_argument("--csv", required=True, help="Ruta al CSV con encabezados")
    pt.add_argument("--target", required=True, help="Nombre de la columna objetivo")
    pt.add_argument("--alpha", type=float, default=0.01, help="Tasa de aprendizaje")
    pt.add_argument("--iters", type=int, default=3000, help="Número de iteraciones GD")
    pt.add_argument("--test-size", type=float, default=0.2, help="Proporción para conjunto de prueba")
    pt.add_argument("--no-intercept", action="store_true", help="Desactiva intercepto")
    pt.add_argument("--no-normalize", action="store_true", help="Desactiva normalización")
    pt.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    pt.set_defaults(func=train_cmd)

    pp = sp.add_parser("predict", help="Usar un modelo guardado para predecir")
    pp.add_argument("--model", default="model.json", help="Ruta al archivo de modelo")
    pp.add_argument("--csv", help="CSV con nuevas observaciones (mismas columnas de features)")
    pp.add_argument("--out", help="Archivo de salida CSV para predicciones")
    pp.add_argument("--take-first", type=int, help="Imprime las primeras N predicciones")
    pp.add_argument("--features", help="Lista separada por comas, ej: '120,3'")
    pp.set_defaults(func=predict_cmd)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
