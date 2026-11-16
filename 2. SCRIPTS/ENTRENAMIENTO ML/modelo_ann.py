import os
import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from evaluar_modelos import guardar_pruebas_hpo, evaluar_y_guardar

# Rutas a la base estandarizada (X_train, X_test, y_train, y_test) generada por base_modelos.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_base")

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))["macroclass_id"]
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))["macroclass_id"]

NOMBRE_MODELO = "ANN"

# Modelo base (perceptron multicapa)
mlp = MLPClassifier(
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
)

# Malla de hiperparametros a explorar (HPO)
# Prueba arquitecturas de 1 y 2 capas ocultas, activaciones, regularizacion L2 (alpha) y tasa de aprendizaje
param_grid = {
    "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.005],
}

# Validacion cruzada estratificada de 3 folds
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Iniciando búsqueda de hiperparámetros (GridSearchCV) para {NOMBRE_MODELO}...")
t0 = time.time()

grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring="f1_macro",  # Prioriza buen desempeño en las 7 clases por igual
    cv=cv_strategy,
    n_jobs=-1,          
    verbose=2,
)

grid_search.fit(X_train, y_train)
print(f"\nEntrenamiento HPO completado en {(time.time() - t0)/60:.2f} minutos.")

# Guardar historial de todas las combinaciones probadas en 3. RESULTADOS/ANN/
guardar_pruebas_hpo(NOMBRE_MODELO, grid_search)

# Evaluar el mejor modelo sobre el conjunto de Test (30%) y guardar entregables
mejor_ann = grid_search.best_estimator_
y_pred = mejor_ann.predict(X_test)
evaluar_y_guardar(NOMBRE_MODELO, mejor_ann, y_test, y_pred)