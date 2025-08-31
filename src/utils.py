import csv
import random
from typing import List, Tuple

def read_csv_xy(path: str, target: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("El CSV no tiene encabezados.")
        if target not in headers:
            raise ValueError(f"La columna objetivo '{target}' no existe. Columnas: {headers}")
        X, y = [], []
        feature_names = [h for h in headers if h != target]
        for row in reader:
            try:
                y_val = float(row[target])
                x_row = [float(row[h]) for h in feature_names]
            except ValueError as e:
                raise ValueError(f"Fila con datos no numéricos: {row}") from e
            X.append(x_row)
            y.append(y_val)
    return X, y, feature_names

def train_test_split(X: list, y: list, test_size: float = 0.2, seed=None):
    assert 0.0 < test_size < 1.0, "test_size debe estar entre 0 y 1."
    n = len(X)
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    cut = int(n * (1.0 - test_size))
    train_idx, test_idx = idx[:cut], idx[cut:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

def write_predictions_csv(path: str, y_true: list, y_pred: list):
    headers = ["y_true", "y_pred"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for t, p in zip(y_true, y_pred):
            writer.writerow([t, p])
