import os
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from evaluar_modelos import guardar_pruebas_hpo, evaluar_y_guardar

# Rutas a la base estandarizada (X_train, X_test, y_train, y_test) generada por base_modelos.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_base")

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))["macroclass_id"]
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["macroclass_id"]

NOMBRE_MODELO = "SVM"

# Modelo base (Support Vector Machine)
svm = SVC(
    cache_size=1000,
    random_state=42,
)

# Malla de hiperparametros a explorar (HPO)
# Explora distintos kernels (RBF, Polinomial, Lineal), penalizacion (C), coeficiente de kernel (gamma),
# grado del polinomio (degree) y balanceo de pesos por clase (class_weight)
param_grid = [
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "class_weight": [None, "balanced"],
    },
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10],
        "degree": [2, 3],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"],
    },
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    },
]

# Validacion cruzada estratificada de 3 folds
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Iniciando búsqueda de hiperparámetros (GridSearchCV) para {NOMBRE_MODELO}...")
t0 = time.time()

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring="f1_macro",  # Pondera todas las macroclases por igual
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)
print(f"\nEntrenamiento HPO completado en {(time.time() - t0)/60:.2f} minutos.")

# Guardar historial de todas las combinaciones probadas en 3. RESULTADOS/SVM/
guardar_pruebas_hpo(NOMBRE_MODELO, grid_search)

# Evaluar el mejor modelo sobre el conjunto de Test (30%) y guardar entregables
mejor_svm = grid_search.best_estimator_
y_pred = mejor_svm.predict(X_test)
evaluar_y_guardar(NOMBRE_MODELO, mejor_svm, y_test, y_pred)
