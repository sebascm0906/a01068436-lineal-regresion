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

Entrenamiento de un modelo

Ejemplo con el dataset incluido (data/toy_data_complex.csv), donde la columna objetivo es price.

Entrenamiento normal (sin regularización)
python src/cli.py train \
  --csv data/toy_data_complex.csv \
  --target price \
  --alpha 0.01 \
  --iters 4000 \
  --test-size 0.2 \
  --seed 42

Entrenamiento con regularización Ridge
python src/cli.py train \
  --csv data/toy_data_complex.csv \
  --target price \
  --alpha 0.01 \
  --iters 4000 \
  --ridge 5.0 \
  --test-size 0.2 \
  --seed 42


Salida esperada:

Métricas en consola (MSE, MAE, R² en train/test).

Archivo model.json con parámetros, escalador y resultados.

Archivo predicciones_test.csv con predicciones sobre el conjunto de prueba.

Predicción con un modelo entrenado

Con un CSV de nuevas observaciones (mismas columnas de entrada que el entrenamiento, sin price):

python src/cli.py predict --model model.json --csv data/toy_data_complex.csv --take-first 5


O pasando características directamente:

python src/cli.py predict --model model.json --features 120,3,2,15,1,10


Debe pasarse el mismo número de características usado en el entrenamiento y en el mismo orden.

Generación de gráficas
1. Scatter, residuos y métricas (modelo normal)
python analisis_modelo.py


Genera:

graf_reales_vs_pred.png

graf_residuos.png

graf_metricas.png

2. Curva de pérdida
python visualizar_modelo.py


Genera:

graf_loss.png

3. Comparación Normal vs Ridge
python analisis_modelo_ridge.py


Genera:

comp_MSE.png

comp_MAE.png

comp_R2.png

Contenido de la implementación

Normalización opcional de características (media y desviación estándar).

Intercepto opcional.

Gradiente descendente batch implementado desde cero.

Métricas: MSE, MAE, R².

Regularización L2 (Ridge) opcional.

Historial de pérdida guardado en model.json.

CLI para entrenamiento y predicción.

Dataset sintético de ejemplo (toy_data_complex.csv).

Scripts de análisis con gráficas listas para incluir en reportes.

Flujo sugerido para reportes

Entrenar modelo normal y guardar métricas.

Generar gráficas con analisis_modelo.py y visualizar_modelo.py.

Entrenar modelo con --ridge.

Generar comparaciones con analisis_modelo_ridge.py.

Copiar tablas de métricas y pegar gráficas en el reporte.

Redactar diagnóstico:

Bias (sesgo): bajo, medio o alto.

Varianza: baja, media o alta.

Nivel de ajuste: underfit, fit o overfit.

Comparación Normal vs Ridge.