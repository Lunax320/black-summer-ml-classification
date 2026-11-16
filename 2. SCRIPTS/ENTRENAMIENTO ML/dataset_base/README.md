# Carpeta `dataset_base`

Esta carpeta es generada automáticamente por el script `base_modelos.py`. Contiene el conjunto de datos de entrenamiento y prueba ya procesado, dividido y escalado, para que cada modelo de ML (LR, DT, SVM, ANN, KNN) sea entrenado de forma estandarizada.

---

## Archivos raw y archivos escalados

- Los modelos basados en distancias o gradientes (LR, SVM, ANN, KNN) necesitan los datos escalados (`X_train.csv`). Si se les pasa los valores crudos (donde las bandas tienen escalas muy distintas), el modelo fallará o rendirá muy mal.
- Los modelos basados en árboles (Decision Tree) funcionan mejor con los datos crudos (`X_train_raw.csv`). Los árboles hacen cortes (ej. "si B08 > 3000"), y es mucho más fácil de interpretar con los valores reales de reflectancia que con valores estandarizados.

---

| Archivo                      | Contenido                                                                                                    | Modelo que lo usa                                          |
| :--------------------------- | :----------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| **`X_train.csv`**            | Las 10 bandas espectrales (B01 a B12) del 70% de los píxeles, **ya estandarizadas** (media 0, desviación 1). | LR, SVM, ANN, KNN                                          |
| **`X_test.csv`**             | Las 10 bandas espectrales del 30% de los píxeles, **estandarizadas** usando la misma escala que `X_train`.   | LR, SVM, ANN, KNN                                          |
| **`X_train_raw.csv`**        | Las 10 bandas espectrales del 70% de los píxeles, **sin escalar** (valores originales de reflectancia).      | Decision Tree                                              |
| **`X_test_raw.csv`**         | Las 10 bandas espectrales del 30% de los píxeles, **sin escalar**.                                           | Decision Tree                                              |
| **`y_train.csv`**            | Las etiquetas reales (`macroclass_id`) correspondientes a los píxeles de entrenamiento.                      | **Todos los modelos**                                      |
| **`y_test.csv`**             | Las etiquetas reales (`macroclass_id`) correspondientes a los píxeles de prueba.                             | **Todos los modelos**                                      |
| **`diccionario_clases.csv`** | Tabla de referencia que mapea el ID numérico con el nombre de la clase (ej. `1 = Agua`).                     | Todos (para las matrices de confusión).                    |
| **`scaler.pkl`**             | El modelo `StandardScaler` entrenado (guarda la media y desviación estándar de los datos).                   | **Fase 3** (para escalar el mosaico completo al predecir). |
