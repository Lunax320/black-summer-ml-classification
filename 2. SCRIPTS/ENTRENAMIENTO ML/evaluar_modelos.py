import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../2. SCRIPTS/ENTRENAMIENTO ML
SCRIPTS_DIR = os.path.dirname(BASE_DIR)                    # .../2. SCRIPTS
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)                 # Raiz del proyecto

DATASET_DIR = os.path.join(BASE_DIR, "dataset_base")
RESULTADOS_DIR = os.path.join(PROJECT_DIR, "3. RESULTADOS")

clases_df = pd.read_csv(os.path.join(DATASET_DIR, "diccionario_clases.csv")).sort_values("macroclass_id")
CLASS_IDS = clases_df["macroclass_id"].tolist()
NOMBRES_CLASES = clases_df["macroclass_name"].tolist()


def guardar_pruebas_hpo(nombre_modelo: str, grid_search, top_n: int = 15) -> pd.DataFrame:
    """
    Guarda el historial de hiperparámetros en 3. RESULTADOS/
    """
    nombre_limpio = nombre_modelo.strip()
    carpeta_upper = nombre_limpio.upper()
    archivo_lower = nombre_limpio.lower()

    carpeta_modelo = os.path.join(RESULTADOS_DIR, carpeta_upper)
    os.makedirs(carpeta_modelo, exist_ok=True)

    df_cv = pd.DataFrame(grid_search.cv_results_)
    cols_utiles = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"]
    df_cv = df_cv[cols_utiles].sort_values("rank_test_score")

    ruta_hpo = os.path.join(carpeta_modelo, f"historial_hpo_{archivo_lower}.tsv")
    df_cv.to_csv(ruta_hpo, sep="\t", index=False)

    print(f"\n>>> TOP {min(top_n, len(df_cv))} COMBINACIONES DE HIPERPARÁMETROS ({carpeta_upper}):")
    print(df_cv.head(top_n).to_string(index=False))
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Historial HPO guardado en: {ruta_hpo}")
    return df_cv


def evaluar_y_guardar(nombre_modelo: str, modelo_entrenado, y_test, y_pred) -> pd.DataFrame:
    """
    Evalúa el modelo sobre el conjunto de Test (30%)
    """
    nombre_limpio = nombre_modelo.strip()
    carpeta_upper = nombre_limpio.upper()
    archivo_lower = nombre_limpio.lower()

    carpeta_modelo = os.path.join(RESULTADOS_DIR, carpeta_upper)
    os.makedirs(carpeta_modelo, exist_ok=True)

    # Metricas globales
    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n==================== RESULTADOS TEST: {carpeta_upper} ====================")
    print(f"Overall Accuracy : {oa:.4f} ({oa*100:.2f}%)")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print(f"Precision (Macro): {prec_macro:.4f}")
    print(f"Recall (Macro)   : {rec_macro:.4f}")
    print(f"F1-Score (Macro) : {f1_macro:.4f}")
    print(f"F1-Score (Weight): {f1_weighted:.4f}")
    print("\n--- Reporte Detallado por Clase ---")
    print(
        classification_report(
            y_test, y_pred, labels=CLASS_IDS, target_names=NOMBRES_CLASES, digits=4, zero_division=0
        )
    )

    # Fila resumen
    resumen = pd.DataFrame([{
        "Modelo": carpeta_upper,
        "Overall_Accuracy": round(oa, 4),
        "Kappa": round(kappa, 4),
        "Precision_Macro": round(prec_macro, 4),
        "Recall_Macro": round(rec_macro, 4),
        "F1_Macro": round(f1_macro, 4),
        "F1_Weighted": round(f1_weighted, 4),
    }])
    resumen.to_csv(os.path.join(carpeta_modelo, f"metricas_{archivo_lower}.tsv"), sep="\t", index=False)

    # Reporte por clase en TSV
    rep_dict = classification_report(
        y_test, y_pred, labels=CLASS_IDS, target_names=NOMBRES_CLASES, output_dict=True, zero_division=0
    )
    df_rep = pd.DataFrame(rep_dict).T.round(4)
    df_rep.index.name = "Clase"
    df_rep.to_csv(os.path.join(carpeta_modelo, f"reporte_clases_{archivo_lower}.tsv"), sep="\t")

    # Matriz de confusion
    ETIQUETAS_CORTAS = {
        1: "Agua",
        2: "Urbanización",
        3: "Pasto",
        4: "Veg. Boscosa\nSana",
        5: "Veg. Boscosa\nQuemada",
        6: "Nubes",
        7: "Sombras",
    }
    labels_visuales = [ETIQUETAS_CORTAS.get(cid, str(cid)) for cid in CLASS_IDS]

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_IDS)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_visuales)
    disp.plot(cmap="Blues", ax=ax, values_format="d")

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    ax.set_xlabel("Clase Predicha", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Clase Real", fontsize=11, fontweight="bold", labelpad=10)
    plt.title(f"Matriz de Confusión (Test) - {carpeta_upper}", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_modelo, f"matriz_confusion_{archivo_lower}.png"), dpi=300)
    plt.close()

    # Guardar modelo entrenado (.pkl)
    joblib.dump(modelo_entrenado, os.path.join(carpeta_modelo, f"modelo_{archivo_lower}.pkl"))

    print(f"Archivos guardados exitosamente en: {carpeta_modelo}")
    return resumen