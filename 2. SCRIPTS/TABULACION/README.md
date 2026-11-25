# Tabulación del mosaico Sentinel-2

Esta carpeta contiene el script de extracción de píxeles y la tabla de entrenamiento utilizada para clasificar coberturas en Blue Mountains tras el evento Black Summer.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `generar_tabulaciones.py` | Extrae los valores espectrales de los píxeles en los polígonos de entrenamiento y exporta la tabla de píxeles válidos del mosaico completo. |
| `AfterProy2_Training_Tabulado.tsv` | Tabla de entrenamiento con **70.519 píxeles etiquetados**. Se usa como entrada de `base_modelos.py` para preparar los conjuntos de entrenamiento y prueba. |
| `Mosaico_Completo_Tabulado.tsv` | Tabla con los **16.071.574 píxeles válidos** del mosaico, sus coordenadas y los valores de las diez bandas espectrales. |

El archivo `Mosaico_Completo_Tabulado.tsv` ocupa aproximadamente **1,32 GB**, por esta razón se encuentra en Google Drive.

🔗 [Tabulación mosaico completo](https://drive.google.com/drive/folders/1xIBJCgLqRLYNdbRptP4XbYm_81qgUfJ3?usp=sharing)

## Estructura de las tablas

Cada fila representa un píxel. Ambas tablas registran su posición en el ráster (`row`, `col`), las coordenadas proyectadas de su centro (`x`, `y`) en **EPSG:7856** y sus diez valores espectrales. La resolución del mosaico es de **20 m**.

La tabla de entrenamiento incluye `macroclass_id` y `macroclass_name` con las siguentes cuatro macroclases

| ID | Macroclase |
| --- | --- |
| `1` | Agua |
| `2` | Urbanización |
| `4` | Vegetación boscosa sana |
| `5` | Vegetación boscosa quemada |
