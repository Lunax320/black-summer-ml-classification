# Black Summer ML Classification — Blue Mountains (2019-2020)

Clasificación supervisada de coberturas terrestres en el Parque Nacional Blue Mountains tras el evento **Black Summer** (Australia, 2019-2020). Este proyecto compara cinco arquitecturas de Machine Learning utilizando datos multiespectrales de Sentinel-2.

---

## Integrantes

- Sara Castro
- Maria Alejandra Garcia
- Juan David Ortiz
- Eliana Pardo
- Ana Maria Romero

**Curso:** Tecnologías Emergentes / GeoAI
**Docente:** Andrés Oswaldo Calderón Romero, Ph.D.

---

## Descripción del Proyecto

El propósito de este proyecto es desarrollar y comparar cinco arquitecturas de clasificación supervisada (Logistic Regression, Decision Trees, SVM, ANN y KNN) para generar un mapa temático de coberturas terrestres. A través de la extracción de datos píxel a píxel (Tabla TSV) y un proceso exhaustivo de optimización de hiperparámetros (HPO), se busca identificar el modelo más robusto para transformar datos de reflectancia multiespectral en inteligencia ambiental accionable.

---

## Datos Utilizados (Fase Post-incendio)

| Parámetro                | Especificación                            |
| ------------------------ | ----------------------------------------- |
| Misión                   | Sentinel-2 MSI (MultiSpectral Instrument) |
| Nivel de procesamiento   | Level-2A (BOA - Bottom-Of-Atmosphere)     |
| Teselas MGRS             | T56HKH y T56HKJ                           |
| Fecha de análisis        | 19 de febrero de 2020 (Post-incendio)     |
| Bandas utilizadas (10)   | B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 |
| Resolución espacial      | Unificada a 20 m (Remuestreo)             |
| Proyección               | GDA2020 / MGA zone 56 (EPSG:7856)         |
| Plataforma de descarga   | Copernicus Data Space Ecosystem           |

---

## Acceso a los Datos y Código

Las capas resultantes (Ráster de predicción final, tablas TSV y resultados de matrices de confusión) se encuentran alojadas en Google Drive debido a su tamaño.

🔗 **[Descargar Resultados y Datos del Proyecto](https://drive.google.com/drive/folders/1NzLZV38xzNx89CI2KvzmwsXG0i_hOacK?usp=sharing)**

---
