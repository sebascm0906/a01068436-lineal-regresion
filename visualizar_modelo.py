import json
import matplotlib.pyplot as plt

# ================================
# Cargar modelo y métricas
# ================================
with open("model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

w = data["model"]["w"]
b = data["model"]["b"]
alpha = data["model"]["alpha"]
iters = data["model"]["iters"]
l2 = data["model"].get("l2", 0.0)
history = data["model"].get("history", [])
results = data.get("results", {})

print("=== Resumen del modelo ===")
print(f"Pesos: {w}")
print(f"Intercepto: {b:.4f}")
print(f"Alpha: {alpha}, Iteraciones: {iters}, L2 (Ridge): {l2}")
print("\n=== Métricas guardadas ===")
for split, vals in results.items():
    print(f"{split}: MSE={vals['MSE']:.2f}, MAE={vals['MAE']:.2f}, R²={vals['R2']:.3f}")

# ================================
# Graficar historial de pérdida
# ================================
if history:
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(history)+1), history, label="MSE")
    plt.xlabel("Iteraciones")
    plt.ylabel("MSE")
    plt.title("Historial de pérdida (Gradiente Descendente)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graf_loss.png")
    plt.show()
    print("\nGráfica guardada como 'graf_loss.png'")
else:
    print("\n⚠️ Este modelo no tiene historial de pérdida guardado.")
