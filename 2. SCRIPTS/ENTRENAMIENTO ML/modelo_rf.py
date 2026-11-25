import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from evaluar_modelos import DATASET_DIR, RESULTADOS_DIR as RESULTADOS_MODELOS, guardar_pruebas_hpo, evaluar_y_guardar

RESULTADOS_DIR = os.path.join(RESULTADOS_MODELOS, "RF")

os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Cargamos datos crudos (raw)
X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test_raw.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))["macroclass_id"]
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["macroclass_id"]

NOMBRE_MODELO = "RF"

print("==========================================================")
print("   ENTRENAMIENTO Y OPTIMIZACIÓN: RANDOM FOREST (BONUS)")
print("==========================================================")

# Modelo base de Ensamble
rf = RandomForestClassifier(random_state=42)

# Malla de hiperparametros para HPO
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}

# Validacion cruzada estratificada de 3 folds
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Iniciando búsqueda de hiperparámetros (GridSearchCV) para {NOMBRE_MODELO}...")
t0 = time.time()

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)
duracion = (time.time() - t0) / 60
print(f"\nEntrenamiento HPO completado en {duracion:.2f} minutos.")
print(f"Mejores Hiperparámetros encontrados: {grid_search.best_params_}")
print(f"Mejor F1-Macro CV: {grid_search.best_score_:.4f}")

# Guardar historial HPO
guardar_pruebas_hpo(NOMBRE_MODELO, grid_search)

# Evaluar sobre test y guardar metricas y matriz de confusion
mejor_rf = grid_search.best_estimator_
y_pred = mejor_rf.predict(X_test)
resumen = evaluar_y_guardar(NOMBRE_MODELO, mejor_rf, y_test, y_pred)

# -------------------------------------------------------------
# ANÁLISIS DE IMPORTANCIA DE BANDAS (FEATURE IMPORTANCE EN RF)
# -------------------------------------------------------------
print("\n>>> Analizando la importancia de las bandas espectrales en Random Forest...")
bandas = list(X_train.columns)
importancias = mejor_rf.feature_importances_

df_importancia = pd.DataFrame({
    "Banda": bandas,
    "Importancia": importancias,
    "Porcentaje": importancias * 100
}).sort_values(by="Importancia", ascending=False).reset_index(drop=True)

# Guardar TSV de importancias
ruta_tsv_imp = os.path.join(RESULTADOS_DIR, "importancia_bandas_rf.tsv")
df_importancia.to_csv(ruta_tsv_imp, sep="\t", index=False)
print(df_importancia.to_string(index=False))
print(f"Importancia de bandas guardada en: {ruta_tsv_imp}")

# Graficar importancia de bandas
plt.figure(figsize=(9, 5))
colores = plt.cm.magma(np.linspace(0.2, 0.85, len(df_importancia)))
barras = plt.bar(df_importancia["Banda"], df_importancia["Porcentaje"], color=colores, edgecolor="black", alpha=0.85)

for bar in barras:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.title("Influencia de Bandas Espectrales (Random Forest - Ensamble)", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Bandas Espectrales Sentinel-2", fontsize=11, fontweight="bold")
plt.ylabel("Importancia Relativa (%)", fontsize=11, fontweight="bold")
plt.ylim(0, max(df_importancia["Porcentaje"]) * 1.15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
ruta_png_imp = os.path.join(RESULTADOS_DIR, "importancia_bandas_rf.png")
plt.savefig(ruta_png_imp, dpi=300)
plt.close()
print(f"Gráfica de importancia guardada en: {ruta_png_imp}")

print("\n[OK] Pipeline de Random Forest completado con éxito.")
