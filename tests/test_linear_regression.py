# Prueba simple sin frameworks externos
from src.linear_regression import LinearRegressionGD, StandardScaler
import random
import math

def gen_data(n=200, seed=123):
    rnd = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1 = rnd.uniform(0, 10)
        x2 = rnd.uniform(-5, 5)
        noise = rnd.gauss(0, 1.5)
        yval = 3.5 * x1 + 2.0 * x2 + 10.0 + noise
        X.append([x1, x2])
        y.append(yval)
    return X, y

def run_test():
    X, y = gen_data()
    sc = StandardScaler()
    Xn = sc.fit_transform(X)
    model = LinearRegressionGD(alpha=0.05, iters=3000, fit_intercept=True)
    model.fit(Xn, y)
    yhat = model.predict(Xn)
    # Debe aproximar bien
    mse = model.mse(y, yhat)
    r2 = model.r2(y, yhat)
    print(f"MSE={mse:.3f} R2={r2:.3f}")
    assert r2 > 0.95, "El R2 debería ser alto en datos sintéticos lineales."

if __name__ == "__main__":
    run_test()
