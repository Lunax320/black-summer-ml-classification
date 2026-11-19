import os
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from evaluar_modelos import guardar_pruebas_hpo, evaluar_y_guardar

# Rutas a la base estandarizada (X_train, X_test, y_train, y_test) generada por base_modelos.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_base")

# KNN usa DISTANCIAS, por eso se usan los datos ESCALADOS (X_train.csv / X_test.csv), no los raw
X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))["macroclass_id"]
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["macroclass_id"]

NOMBRE_MODELO = "KNN"

# Modelo base (K-Nearest Neighbors)
knn = KNeighborsClassifier(n_jobs=-1)

# Malla de hiperparametros a explorar (HPO)
# n_neighbors: cuantos vecinos votan (k)
# weights: 'uniform' = todos los vecinos pesan igual; 'distance' = los mas cercanos pesan mas
# metric / p: tipo de distancia (Manhattan p=1, Euclidiana p=2, Minkowski p=3)
param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
    "weights": ["uniform", "distance"],
    "metric": ["minkowski"],
    "p": [1, 2, 3],
}

# Validacion cruzada estratificada de 3 folds (igual que los demas modelos)
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Iniciando búsqueda de hiperparámetros (GridSearchCV) para {NOMBRE_MODELO}...")
t0 = time.time()

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring="f1_macro",  # Prioriza buen desempeño en las 7 clases por igual
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)
print(f"\nEntrenamiento HPO completado en {(time.time() - t0)/60:.2f} minutos.")

# Guardar historial de todas las combinaciones probadas en 3. RESULTADOS/KNN/
guardar_pruebas_hpo(NOMBRE_MODELO, grid_search)

# Evaluar el mejor modelo sobre el conjunto de Test (30%) y guardar entregables
mejor_knn = grid_search.best_estimator_
y_pred = mejor_knn.predict(X_test)
evaluar_y_guardar(NOMBRE_MODELO, mejor_knn, y_test, y_pred)
