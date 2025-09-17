import json
import math
from typing import List, Dict, Optional

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
            s = sum(X[i][j] for i in range(n))
            mu = s / n
            self.means[j] = mu
            var = sum((X[i][j] - mu) ** 2 for i in range(n)) / n
            self.stds[j] = math.sqrt(var) if var > 0 else 1.0
        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        if self.means is None or self.stds is None:
            raise RuntimeError("StandardScaler no está ajustado.")
        n = len(X)
        n_features = len(self.means)
        return [[(X[i][j] - self.means[j]) / self.stds[j] for j in range(n_features)] for i in range(n)]

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
    def __init__(self, alpha: float = 0.01, iters: int = 1000,
                 fit_intercept: bool = True, l2: float = 0.0):
        """
        Regresión Lineal con Gradiente Descendente.
        - alpha: tasa de aprendizaje
        - iters: número de iteraciones
        - fit_intercept: si incluye bias
        - l2: coeficiente de regularización L2 (0.0 = sin regularización)
        """
        self.alpha = alpha
        self.iters = iters
        self.fit_intercept = fit_intercept
        self.l2 = l2
        self.w: Optional[List[float]] = None
        self.b: float = 0.0
        self.history: List[float] = []

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
        self.history = []

        for it in range(self.iters):
            grad_w = [0.0] * m
            grad_b = 0.0
            for i in range(n):
                y_hat = self._predict_row(X[i])
                err = y_hat - y[i]
                if self.fit_intercept:
                    grad_b += err
                for j in range(m):
                    grad_w[j] += err * X[i][j]
            # aplicar promedio + regularización
            lr = self.alpha / n
            for j in range(m):
                grad_w[j] += self.l2 * self.w[j]  # penalización ridge
                self.w[j] -= lr * grad_w[j]
            if self.fit_intercept:
                self.b -= lr * grad_b

            # guardar MSE en historial
            yhat = [self._predict_row(xi) for xi in X]
            mse_val = self.mse(y, yhat)
            self.history.append(mse_val)

        return self

    @staticmethod
    def mse(y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n if n else float("nan")

    @staticmethod
    def mae(y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n if n else float("nan")

    @staticmethod
    def r2(y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        if n == 0:
            return float("nan")
        mu = sum(y_true) / n
        ss_tot = sum((yi - mu) ** 2 for yi in y_true)
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def save_model(path: str, model: LinearRegressionGD,
               scaler: Optional[StandardScaler],
               feature_names: list, config: dict):
    payload = {
        "w": model.w,
        "b": model.b,
        "fit_intercept": model.fit_intercept,
        "alpha": model.alpha,
        "iters": model.iters,
        "l2": model.l2,
        "history": model.history,
        "scaler": scaler.to_dict() if scaler is not None else None,
        "feature_names": feature_names,
        "config": config,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    model = LinearRegressionGD(
        alpha=payload.get("alpha", 0.01),
        iters=payload.get("iters", 1000),
        fit_intercept=payload.get("fit_intercept", True),
        l2=payload.get("l2", 0.0),
    )
    model.w = payload["w"]
    model.b = payload["b"]
    model.history = payload.get("history", [])
    scaler = StandardScaler.from_dict(payload["scaler"]) if payload["scaler"] is not None else None
    feature_names = payload.get("feature_names", None)
    return model, scaler, feature_names
