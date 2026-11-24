import os
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from evaluar_modelos import guardar_pruebas_hpo, evaluar_y_guardar

# Rutas a la base estandarizada (X_train, X_test, y_train, y_test) generada por base_modelos.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_base")

# LR es un modelo basado en gradientes: usa los datos ESCALADOS (X_train.csv / X_test.csv)
X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))["macroclass_id"]
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["macroclass_id"]

NOMBRE_MODELO = "LR"

# Modelo base (Regresion Logistica multiclase)
lr = LogisticRegression(
    max_iter=5000,
    random_state=42,
)

# Malla de hiperparametros a explorar (HPO)
# C: inverso de la regularizacion L2 (menor C = regularizacion mas fuerte, modelo mas simple)
# solver: algoritmo de optimizacion (lbfgs y newton-cg soportan multiclase con regularizacion L2)
# class_weight: balancea el peso de las clases si estan desbalanceadas
# Nota: no se usa solver "saga" (L1/ElasticNet) porque es muy lento con ~50 mil pixeles.
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    "solver": ["lbfgs", "newton-cg"],
    "class_weight": [None, "balanced"],
}

# Validacion cruzada estratificada de 3 folds (igual que SVM para comparar en igualdad de condiciones)
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Iniciando búsqueda de hiperparámetros (GridSearchCV) para {NOMBRE_MODELO}...")
t0 = time.time()

grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    scoring="f1_macro",  # Prioriza buen desempeño en las 7 clases por igual
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)
print(f"\nEntrenamiento HPO completado en {(time.time() - t0)/60:.2f} minutos.")

# Guardar historial de todas las combinaciones probadas en 3. RESULTADOS/LR/
guardar_pruebas_hpo(NOMBRE_MODELO, grid_search)

# Evaluar el mejor modelo sobre el conjunto de Test (30%) y guardar entregables
mejor_lr = grid_search.best_estimator_
y_pred = mejor_lr.predict(X_test)
evaluar_y_guardar(NOMBRE_MODELO, mejor_lr, y_test, y_pred)
