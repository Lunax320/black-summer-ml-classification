import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from evaluar_modelos import guardar_pruebas_hpo, evaluar_y_guardar

# Rutas a la base generada por base_modelos.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset_base")
RESULTADOS_DIR = os.path.join(PROJECT_DIR, "3. RESULTADOS", "DT")

os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Cargamos datos crudos (raw) para preservar la interpretabilidad fisica de las reflectancias
X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test_raw.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))["macroclass_id"]
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["macroclass_id"]

NOMBRE_MODELO = "DT"

print("==========================================================")
print("   ENTRENAMIENTO Y OPTIMIZACIÓN: DECISION TREES (DT)")
print("==========================================================")

# Modelo base
dt = DecisionTreeClassifier(random_state=42)

# Malla de hiperparametros para HPO
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "class_weight": [None, "balanced"],
}

# Validacion cruzada estratificada de 3 folds
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Iniciando búsqueda de hiperparámetros (GridSearchCV) para {NOMBRE_MODELO}...")
t0 = time.time()

grid_search = GridSearchCV(
    estimator=dt,
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
mejor_dt = grid_search.best_estimator_
y_pred = mejor_dt.predict(X_test)
resumen = evaluar_y_guardar(NOMBRE_MODELO, mejor_dt, y_test, y_pred)

# -------------------------------------------------------------
# ANÁLISIS ESPECÍFICO DE BANDAS (FEATURE IMPORTANCE)
# -------------------------------------------------------------
print("\n>>> Analizando la importancia de las bandas espectrales...")
bandas = list(X_train.columns)
importancias = mejor_dt.feature_importances_

df_importancia = pd.DataFrame({
    "Banda": bandas,
    "Importancia": importancias,
    "Porcentaje": importancias * 100
}).sort_values(by="Importancia", ascending=False).reset_index(drop=True)

# Guardar TSV de importancias
ruta_tsv_imp = os.path.join(RESULTADOS_DIR, "importancia_bandas_dt.tsv")
df_importancia.to_csv(ruta_tsv_imp, sep="\t", index=False)
print(df_importancia.to_string(index=False))
print(f"Importancia de bandas guardada en: {ruta_tsv_imp}")

# Graficar importancia de bandas
plt.figure(figsize=(9, 5))
colores = plt.cm.viridis(np.linspace(0.2, 0.85, len(df_importancia)))
barras = plt.bar(df_importancia["Banda"], df_importancia["Porcentaje"], color=colores, edgecolor="black", alpha=0.85)

for bar in barras:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.title("Influencia de Bandas Espectrales en la Clasificación (Decision Tree)", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Bandas Espectrales Sentinel-2", fontsize=11, fontweight="bold")
plt.ylabel("Importancia Relativa (%)", fontsize=11, fontweight="bold")
plt.ylim(0, max(df_importancia["Porcentaje"]) * 1.15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
ruta_png_imp = os.path.join(RESULTADOS_DIR, "importancia_bandas_dt.png")
plt.savefig(ruta_png_imp, dpi=300)
plt.close()
print(f"Gráfica de importancia guardada en: {ruta_png_imp}")

# -------------------------------------------------------------
# VISUALIZACIÓN DE LA ESTRUCTURA DEL ÁRBOL (PRIMEROS NIVELES)
# -------------------------------------------------------------
clases_df = pd.read_csv(os.path.join(DATASET_DIR, "diccionario_clases.csv")).sort_values("macroclass_id")
nombres_clases = clases_df["macroclass_name"].tolist()

print("\n>>> Generando diagrama esquemático de los primeros niveles del árbol...")
plt.figure(figsize=(20, 10))
plot_tree(
    mejor_dt,
    max_depth=3,
    feature_names=bandas,
    class_names=nombres_clases,
    filled=True,
    rounded=True,
    fontsize=9,
    precision=2
)
plt.title(f"Árbol de Decisión Poda Visual (Profundidad = 3) - Criterio: {mejor_dt.criterion}", fontsize=14, fontweight="bold")
plt.tight_layout()
ruta_arbol = os.path.join(RESULTADOS_DIR, "estructura_arbol_dt.png")
plt.savefig(ruta_arbol, dpi=300)
plt.close()
print(f"Diagrama del árbol guardado en: {ruta_arbol}")

print("\n[OK] Pipeline de Decision Tree completado con éxito.")
