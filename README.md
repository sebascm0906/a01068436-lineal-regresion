# ML desde cero: Regresión Lineal con Gradiente Descendente

Implementación **sin librerías** de un modelo de Regresión Lineal multivariable con entrenamiento por Gradiente Descendente.
Todo en Python estándar (solo `math`, `csv`, `argparse`, etc.). Corre en terminal como un `.py` normal.

## Estructura del repo

```
ml-desde-cero/
├── LICENSE
├── .gitignore
├── README.md
├── data/
│   └── toy_data.csv
├── src/
│   ├── linear_regression.py
│   ├── utils.py
│   └── cli.py
└── tests/
    └── test_linear_regression.py
```

## Requisitos

- Python 3.9+ (o similar)
- No se usan bibliotecas externas.

## Cómo ejecutar un experimento de punta a punta

Ejemplo con el dataset sintético incluido (`data/toy_data.csv`), donde la columna objetivo es `price`:

```bash
python src/cli.py train   --csv data/toy_data.csv   --target price   --alpha 0.01   --iters 4000   --test-size 0.2   --seed 42
```

Salida esperada (resumen): métricas en train y test, primeras predicciones vs valores reales y un archivo `model.json` con los parámetros aprendidos.
Además se genera `predicciones_test.csv` con las predicciones sobre el conjunto de prueba.

### Predecir con un modelo entrenado

Con un CSV de nuevas observaciones (mismas columnas de entrada que el entrenamiento, sin `price`):

```bash
python src/cli.py predict --csv data/toy_data.csv --take-first 5
```

O pasar características “en línea”:

```bash
python src/cli.py predict --features 120,3
```

> El ejemplo anterior asume 2 características. Si tu modelo fue entrenado con 2 columnas de entrada, debes pasar dos números en el mismo orden.


## ¿Qué incluye exactamente la implementación?

- Normalización opcional de características (media y desviación).
- Intercepto opcional.
- Entrenamiento por gradiente descendente batch.
- Métricas: MSE, MAE, R².
- Guardado/carga del modelo en JSON.
- CLI con `train` y `predict`.
- Dataset sintético de ejemplo.
