"""Clasifica Mosaico_Completo.tif con SVM

Coloca el raster junto al script y usa el Python de QGIS para ejecutarlo
"""

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

try:
    from osgeo import gdal
except ModuleNotFoundError as error:
    if error.name != "osgeo":
        raise
    raise SystemExit("Falta GDAL en este Python, usa python-qgis-ltr.bat de QGIS") from None

import joblib
import numpy as np
import pandas as pd

# TABULACION es una carpeta hermana de la que contiene este script
CARPETA_SCRIPT = Path(__file__).resolve().parent
CARPETA_SCRIPTS = CARPETA_SCRIPT.parent
sys.path.insert(0, str(CARPETA_SCRIPTS))
from TABULACION.generar_tabulaciones import BAND_NAMES, NOMBRES_MACROCLASES


FILAS_POR_BLOQUE = 128
NODATA_SALIDA = 0
NOMBRE_TIF = "Raster_Clasificado.tif"
NOMBRE_CSV = "areas_por_macroclase.csv"


def predecir_mosaico(origen, modelo, scaler, ruta_tif: Path) -> tuple[np.ndarray, float]:
    """Escribe una banda de macroclases y devuelve sus conteos y el area por pixel"""
    if origen.RasterCount != len(BAND_NAMES):
        raise ValueError(f"El mosaico debe tener las {len(BAND_NAMES)} bandas del entrenamiento")

    transformacion = origen.GetGeoTransform(can_return_null=True)
    crs = origen.GetSpatialRef()
    if transformacion is None or crs is None or not crs.IsProjected():
        raise ValueError("El mosaico debe tener una transformacion y un CRS proyectado para calcular areas")

    # El determinante de la transformacion da el area de un pixel en unidades del CRS al cuadrado
    # Se convierte primero a metros cuadrados y luego a hectareas
    _, ancho_pixel, giro_x, _, giro_y, alto_pixel = transformacion
    area_pixel_ha = abs(ancho_pixel * alto_pixel - giro_x * giro_y) * crs.GetLinearUnits() ** 2 / 10_000
    if not np.isfinite(area_pixel_ha) or area_pixel_ha <= 0:
        raise ValueError("La transformacion del mosaico no define un area de pixel valida")

    ancho, alto = origen.RasterXSize, origen.RasterYSize
    nodata = origen.GetRasterBand(1).GetNoDataValue()
    if nodata is None:
        nodata = 0  # Mismo valor por defecto que la tabulacion

    # Cada posicion del contador corresponde a un posible codigo de la banda de salida
    conteos = np.zeros(256, dtype=np.int64)
    t0 = perf_counter()
    driver = gdal.GetDriverByName("GTiff")
    with driver.Create(str(ruta_tif), ancho, alto, 1, gdal.GDT_Byte, ["TILED=YES", "COMPRESS=DEFLATE"]) as destino:
        # Copiar la cuadricula y el CRS para que el resultado se alinee con el mosaico original
        destino.SetGeoTransform(transformacion)
        destino.SetSpatialRef(crs)
        destino.SetMetadataItem("MODELO", "SVM")
        banda = destino.GetRasterBand(1)
        banda.SetNoDataValue(NODATA_SALIDA)
        banda.SetDescription("Macroclase")

        # Procesar franjas de filas para limitar el uso de memoria
        for fila in range(0, alto, FILAS_POR_BLOQUE):
            filas = min(FILAS_POR_BLOQUE, alto - fila)
            bloque = origen.ReadAsArray(0, fila, ancho, filas)
            # La tabulacion conserva los pixeles con al menos una banda distinta de NoData
            validos = np.any(bloque != nodata, axis=0) & np.all(np.isfinite(bloque), axis=0)
            clasificado = np.full((filas, ancho), NODATA_SALIDA, dtype=np.uint8)

            if np.any(validos):
                # La tabla de entrenamiento guardo las reflectancias como int32
                reflectancias = pd.DataFrame(bloque[:, validos].T.astype(np.int32), columns=BAND_NAMES)
                # Aplicar el escalador ajustado con el entrenamiento antes de predecir con SVM
                escaladas = pd.DataFrame(scaler.transform(reflectancias), columns=BAND_NAMES)
                predicciones = modelo.predict(escaladas)
                # Colocar cada clase en su posicion original y contarla para la tabla de areas
                clasificado[validos] = predicciones
                conteos += np.bincount(predicciones, minlength=256)

            banda.WriteArray(clasificado, 0, fila)
            print(
                f"Filas {fila + filas:,}/{alto:,} | {conteos.sum():,} pixeles clasificados "
                f"| {(perf_counter() - t0) / 60:.1f} min",
                flush=True,
            )

    if not conteos.any():
        raise ValueError("El mosaico no contiene pixeles validos para clasificar")

    return conteos, area_pixel_ha


def main() -> None:
    gdal.UseExceptions()

    # El raster va junto al script, pero debido al peso se dejo en el drive
    # Si se quiere compilar este script, el archivo del raster completo se debe dejar en esta carpeta
    resultados = CARPETA_SCRIPTS.parent / "3. RESULTADOS"
    base = CARPETA_SCRIPTS / "ENTRENAMIENTO ML" / "dataset_base"
    mosaico = CARPETA_SCRIPT / "Mosaico_Completo.tif"
    if not mosaico.is_file():
        raise FileNotFoundError(f"Coloca Mosaico_Completo.tif al lado del script: {CARPETA_SCRIPT}")

    modelo = joblib.load(resultados / "MODELOS ML" / "SVM" / "modelo_svm.pkl")
    scaler = joblib.load(base / "scaler.pkl")
    clases = pd.read_csv(base / "diccionario_clases.csv").sort_values("macroclass_id")
    ids = clases["macroclass_id"].to_numpy()
    if not np.array_equal(ids, sorted(NOMBRES_MACROCLASES)) or not np.array_equal(modelo.classes_, ids):
        raise ValueError("El SVM y el diccionario deben corresponder a las cuatro macroclases de la Entrega 3")
    if list(scaler.feature_names_in_) != BAND_NAMES or list(modelo.feature_names_in_) != BAND_NAMES:
        raise ValueError("El orden de las bandas de la tabulacion debe coincidir con el escalador y el SVM")

    # Las ejecuciones posteriores usan una carpeta fechada para conservar las salidas anteriores
    carpeta_salida = resultados / "PREDICCION"
    if carpeta_salida.exists():
        carpeta_salida = resultados / f"PREDICCION_{datetime.now():%Y%m%d_%H%M%S_%f}"

    print(f"Clasificando con SVM: {mosaico}", flush=True)
    # Generar ambos archivos juntos al terminar la prediccion
    with TemporaryDirectory(prefix=".prediccion-", dir=resultados) as temp_dir:
        salida = Path(temp_dir) / "salida"
        salida.mkdir()
        with gdal.Open(str(mosaico)) as origen:
            conteos, area_pixel_ha = predecir_mosaico(origen, modelo, scaler, salida / NOMBRE_TIF)

        clases["pixeles"] = conteos[ids]
        clases["area_ha"] = clases["pixeles"] * area_pixel_ha
        clases.to_csv(salida / NOMBRE_CSV, index=False, float_format="%.4f")
        salida.rename(carpeta_salida)

    print(f"\nGeoTIFF: {carpeta_salida / NOMBRE_TIF}")
    print(f"Areas por clase: {carpeta_salida / NOMBRE_CSV}")


if __name__ == "__main__":
    main()
