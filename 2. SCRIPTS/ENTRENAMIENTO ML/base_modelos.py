import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
SCRIPTS_DIR = os.path.dirname(BASE_DIR)               
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)           

# Ruta al archivo tabulado
INPUT_TSV = os.path.join(SCRIPTS_DIR, "TABULACION", "AfterProy2_Training_Tabulado.tsv")

# Ruta de salida
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_base")

# Rutas y parametros ESTANDAR
RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70 train / 30 test

BANDAS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]

# Cargar el TSV tabulado
df = pd.read_csv(INPUT_TSV, sep="\t")
clases_df = df[["macroclass_id", "macroclass_name"]].drop_duplicates().sort_values("macroclass_id")
if len(clases_df) != 4 or not (clases_df.nunique() == 4).all():
    raise ValueError(
        "El TSV debe contener las cuatro macroclases"
    )

# Seleccionar UNICAMENTE las 10 bandas espectrales (X) y la macroclase (y)
X = df[BANDAS].copy()
y = df["macroclass_id"].copy()

# PRIMERO dividir en 70/30 estratificado (evita data leakage al estandarizar)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# SEGUNDO estandarizar (fit SOLO en train, transform en train y test)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_raw),
    columns=BANDAS,
    index=X_train_raw.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_raw),
    columns=BANDAS,
    index=X_test_raw.index
)

# Guardar los datasets estandarizados (LR, SVM, ANN, KNN)
os.makedirs(OUTPUT_DIR, exist_ok=True)
X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)

# Guardar también los datos crudos para DT y RF
X_train_raw.to_csv(os.path.join(OUTPUT_DIR, "X_train_raw.csv"), index=False)
X_test_raw.to_csv(os.path.join(OUTPUT_DIR, "X_test_raw.csv"), index=False)

# Guardar etiquetas (y) y el diccionario de nombres de clases
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

clases_df.to_csv(os.path.join(OUTPUT_DIR, "diccionario_clases.csv"), index=False)

# Guardar el objeto scaler para usarlo en la inferencia del mosaico completo (Esto es para la entrega 3)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print("=== BASE GENERADA EXITOSAMENTE ===")
print(f"Total muestras: {len(df):,}")
print(f"Train (70%): {len(X_train_scaled):,} píxeles | Test (30%): {len(X_test_scaled):,} píxeles")
print(f"Archivos guardados en: ./{OUTPUT_DIR}/")
