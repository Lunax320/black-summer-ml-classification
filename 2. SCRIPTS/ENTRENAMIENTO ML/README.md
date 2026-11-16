# ENTRENAMIENTO ML

Esta carpeta contiene el código para el entrenamiento, evaluación y almacenamiento de los modelos de ML del proyecto.

Para mantener la consistencia en los datos y en la evaluación de todos los modelos, se han creado dos scripts base (`base_modelos.py` y `evaluar_modelos.py`).

### 1. `base_modelos.py`
Este script es el encargado de preparar los datos para todos los modelos. Lee el archivo tabulado original (`.tsv`), separa las 10 bandas espectrales (X) de la variable objetivo (y), y realiza la división de los datos en 70% entrenamiento y 30% prueba. Para la prevención de data leakage, estandariza los datos *después* de la división. Ajusta el escalador (`StandardScaler`) únicamente con el conjunto de entrenamiento y luego lo aplica al de prueba. Como resultado, crea la carpeta `dataset_base/` con los archivos `X_train.csv`, `X_test.csv` (escalados), `X_train_raw.csv`, `X_test_raw.csv` (crudos para modelos que lo requieran) y el archivo `scaler.pkl`.

### 2. `evaluar_modelos.py`

Contiene dos funciones diseñadas para automatizar el análisis de los resultados. La primera es `guardar_pruebas_hpo(nombre_modelo, grid_search)`: recibe el objeto de búsqueda de hiperparámetros (`GridSearchCV`). Extrae la tabla interna con todas las combinaciones evaluadas (`cv_results_`), la ordena por rendimiento (`rank_test_score`) y la exporta como un archivo `.tsv`. Su utilidad es generar automáticamente el registro histórico del *tuning*, demostrando qué hiperparámetros se probaron y cuáles dieron los mejores resultados (los cuales también imprime en consola).

La segunda es `evaluar_y_guardar(nombre_modelo, modelo_entrenado, y_test, y_pred)`: compara las predicciones del modelo (`y_pred`) contra las etiquetas reales (`y_test`). Calcula métricas globales (Accuracy, Kappa, Precisión, Recall, F1-Score) y métricas por clase. Genera y guarda una imagen de la matriz de confusión y exporta los resultados en formato `.tsv`. Su utilidad es estandarizar el reporte de calidad para todos los modelos.

---
    
### **Scripts de modelos**

En esta carpeta también se alojan los scripts individuales de cada modelo. Todos estos scripts están estructurados para importar y utilizar las funciones de `evaluar_modelos.py`, con el fin de generar los resultados y métricas bajo el mismo estándar.