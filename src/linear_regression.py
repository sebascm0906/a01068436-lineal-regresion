import json
import math
from typing import List, Dict

class StandardScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X: List[List[float]]):
        n_features = len(X[0])
        self.means = [0.0] * n_features
        self.stds = [0.0] * n_features
        n = len(X)
        for j in range(n_features):
            s = 0.0
            for i in range(n):
                s += X[i][j]
            mu = s / n
            self.means[j] = mu
            # std
            var = 0.0
            for i in range(n):
                d = X[i][j] - mu
                var += d * d
            var /= n
            self.stds[j] = math.sqrt(var) if var > 0 else 1.0
        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        if self.means is None or self.stds is None:
            raise RuntimeError("StandardScaler no está ajustado.")
        n = len(X)
        n_features = len(self.means)
        X_out = [[0.0] * n_features for _ in range(n)]
        for i in range(n):
            for j in range(n_features):
                X_out[i][j] = (X[i][j] - self.means[j]) / self.stds[j]
        return X_out

    def fit_transform(self, X: List[List[float]]):
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> Dict:
        return {"means": self.means, "stds": self.stds}

    @staticmethod
    def from_dict(d: Dict):
        sc = StandardScaler()
        sc.means = d["means"]
        sc.stds = d["stds"]
        return sc

class LinearRegressionGD:
    def __init__(self, alpha: float = 0.01, iters: int = 1000, fit_intercept: bool = True):
        self.alpha = alpha
        self.iters = iters
        self.fit_intercept = fit_intercept
        self.w = None  # pesos
        self.b = 0.0   # intercepto

    def _predict_row(self, x_row: List[float]) -> float:
        s = self.b if self.fit_intercept else 0.0
        for j, wj in enumerate(self.w):
            s += wj * x_row[j]
        return s

    def predict(self, X: List[List[float]]) -> List[float]:
        if self.w is None:
            raise RuntimeError("El modelo no está entrenado.")
        return [self._predict_row(r) for r in X]

    def fit(self, X: List[List[float]], y: List[float]):
        n = len(X)
        if n == 0:
            raise ValueError("Datos vacíos.")
        m = len(X[0])  # número de características
        self.w = [0.0] * m
        self.b = 0.0
        for it in range(self.iters):
            # Gradientes
            grad_w = [0.0] * m
            grad_b = 0.0
            for i in range(n):
                y_hat = self._predict_row(X[i])
                err = y_hat - y[i]
                if self.fit_intercept:
                    grad_b += err
                for j in range(m):
                    grad_w[j] += err * X[i][j]
            # Promedio
            lr = self.alpha / n
            for j in range(m):
                self.w[j] -= lr * grad_w[j]
            if self.fit_intercept:
                self.b -= lr * grad_b
        return self

    @staticmethod
    def mse(y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        s = 0.0
        for i in range(n):
            d = y_true[i] - y_pred[i]
            s += d * d
        return s / n if n else float("nan")

    @staticmethod
    def mae(y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        s = 0.0
        for i in range(n):
            s += abs(y_true[i] - y_pred[i])
        return s / n if n else float("nan")

    @staticmethod
    def r2(y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        if n == 0:
            return float("nan")
        mu = sum(y_true) / n
        ss_tot = 0.0
        ss_res = 0.0
        for i in range(n):
            d = y_true[i] - mu
            ss_tot += d * d
            e = y_true[i] - y_pred[i]
            ss_res += e * e
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def save_model(path: str, model: LinearRegressionGD, scaler: StandardScaler, feature_names: list, config: dict):
    payload = {
        "w": model.w,
        "b": model.b,
        "fit_intercept": model.fit_intercept,
        "alpha": model.alpha,
        "iters": model.iters,
        "scaler": scaler.to_dict() if scaler is not None else None,
        "feature_names": feature_names,
        "config": config,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_model(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    model = LinearRegressionGD(alpha=payload["alpha"], iters=payload["iters"], fit_intercept=payload["fit_intercept"])
    model.w = payload["w"]
    model.b = payload["b"]
    scaler = StandardScaler.from_dict(payload["scaler"]) if payload["scaler"] is not None else None
    feature_names = payload.get("feature_names", None)
    return model, scaler, feature_names
