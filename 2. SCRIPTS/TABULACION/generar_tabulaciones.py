"""
Script para generar las tabulaciones de píxeles (en formato TSV) a partir de:
1. Mosaico_Completo.tif (Imagen satelital Sentinel-2 de 10 bandas a 20m, EPSG:7856)
2. AfterProy2.scpx (Training input de Semi-Automatic Classification Plugin en QGIS)

Archivos generados:
- AfterProy2_Training_Tabulado.tsv: Píxeles de entrenamiento con coordenadas, valores de las 10 bandas,
  macroclass_id y macroclass_name de la Entrega 3. Usa --desde-tsv para actualizar un TSV existente.
- Mosaico_Completo_Tabulado.tsv: Todos los píxeles válidos (sin NoData=0) del mosaico con coordenadas
  (row, col, x, y) y valores de las 10 bandas para predicción y reconstrucción del mapa clasificado.
"""

import argparse
import os
import struct
import sqlite3
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd

# Nombres de las 10 bandas del mosaico Sentinel-2
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]

NOMBRES_MACROCLASES = {
    1: "Agua",
    2: "Urbanización",
    4: "Vegetación boscosa sana",
    5: "Vegetación boscosa quemada",
}
FUSIONES_MACROCLASES = {3: 4, 6: 2}  # Pasto → Vegetación boscosa sana, Nubes → Urbanización
ID_SOMBRAS = 7


def guardar_tsv_entrenamiento(df: pd.DataFrame, output_tsv: str) -> pd.DataFrame:
    """Aplica las macroclases de la Entrega 3 y guarda el TSV de entrenamiento."""
    df_train = df.loc[~df["macroclass_id"].isin([ID_SOMBRAS])].copy()
    ids = df_train["macroclass_id"].replace(FUSIONES_MACROCLASES)
    nombres = ids.map(NOMBRES_MACROCLASES)
    if nombres.isna().any():
        ids_desconocidos = ids[nombres.isna()].unique().tolist()
        raise ValueError(f"IDs de macroclase desconocidos: {ids_desconocidos}")
    if df_train.empty:
        raise ValueError("No quedan píxeles de entrenamiento después de excluir Sombras.")

    df_train["macroclass_id"] = ids.astype(int)
    df_train["macroclass_name"] = nombres

    # Reemplazar el TSV solo cuando su escritura haya terminado.
    output_dir = os.path.dirname(os.path.abspath(output_tsv))
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        temp_tsv = os.path.join(temp_dir, os.path.basename(output_tsv))
        df_train.to_csv(temp_tsv, sep="\t", index=False, lineterminator="\n")
        os.replace(temp_tsv, output_tsv)

    print(f"      Archivo generado: {output_tsv}")
    print(f"      Total píxeles de entrenamiento: {len(df_train):,}")
    print("      Distribución por macroclase:")
    for mc_id, count in df_train["macroclass_id"].value_counts().sort_index().items():
        print(f"        - [{mc_id}] {NOMBRES_MACROCLASES[mc_id]}: {count:,} píxeles")

    return df_train


def read_geotiff_metadata(tif_path: str) -> dict:
    """Lee las etiquetas esenciales del GeoTIFF (dimensiones, georreferenciación y offset de datos)."""
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    type_fmt = {1: "B", 2: "s", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i", 11: "f", 12: "d"}

    with open(tif_path, "rb") as f:
        magic, version, ifd_offset = struct.unpack("<2sHI", f.read(8))
        if magic != b"II" or version != 42:
            raise ValueError(f"Formato TIFF no soportado: {magic}, version={version}")

        f.seek(ifd_offset)
        num_tags = struct.unpack("<H", f.read(2))[0]
        raw_entries = [struct.unpack("<HHII", f.read(12)) for _ in range(num_tags)]

        tags = {}
        for tag, ttype, count, val_off in raw_entries:
            sz = type_sizes.get(ttype, 1) * count
            if sz <= 4:
                raw = struct.pack("<I", val_off)[:sz]
            else:
                f.seek(val_off)
                raw = f.read(sz)
            if ttype == 2:
                tags[tag] = raw.decode("utf-8", errors="replace").rstrip("\x00")
            elif ttype in type_fmt:
                tags[tag] = struct.unpack(f"<{count}{type_fmt[ttype]}", raw)

    width = tags[256][0]
    height = tags[257][0]
    num_bands = tags[277][0]
    strip_offsets = tags[273]
    pixel_scale = tags[33550]  # (dx, dy, dz)
    tiepoint = tags[33922]     # (i, j, k, x0, y0, z0)
    nodata_str = tags.get(42113, "0")

    return {
        "width": width,
        "height": height,
        "num_bands": num_bands,
        "data_offset": strip_offsets[0],
        "dx": float(pixel_scale[0]),
        "dy": float(pixel_scale[1]),
        "x0": float(tiepoint[3]),
        "y0": float(tiepoint[4]),
        "nodata": float(nodata_str),
    }


def parse_gpkg_geom(geom_bytes: bytes) -> list:
    """Decodifica geometrías WKB (Polygon y MultiPolygon) desde un blob GeoPackage."""
    magic, ver, flags, srs_id = struct.unpack("<2sBBi", geom_bytes[:8])
    env_type = (flags >> 1) & 0x07
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    wkb = geom_bytes[8 + env_sizes[env_type]:]
    bo, wkb_type = struct.unpack("<BI", wkb[:5])
    polys = []
    off = 5

    if wkb_type == 3:  # Polygon
        num_rings = struct.unpack("<I", wkb[off:off + 4])[0]
        off += 4
        rings = []
        for _ in range(num_rings):
            num_pts = struct.unpack("<I", wkb[off:off + 4])[0]
            off += 4
            pts = np.frombuffer(wkb, dtype="<f8", count=num_pts * 2, offset=off).reshape(-1, 2)
            off += num_pts * 16
            rings.append(pts)
        polys.append(rings)
    elif wkb_type == 6:  # MultiPolygon
        num_polys = struct.unpack("<I", wkb[off:off + 4])[0]
        off += 4
        for _ in range(num_polys):
            bo_p, p_type, num_rings = struct.unpack("<BII", wkb[off:off + 9])
            off += 9
            rings = []
            for _ in range(num_rings):
                num_pts = struct.unpack("<I", wkb[off:off + 4])[0]
                off += 4
                pts = np.frombuffer(wkb, dtype="<f8", count=num_pts * 2, offset=off).reshape(-1, 2)
                off += num_pts * 16
                rings.append(pts)
            polys.append(rings)
    else:
        raise ValueError(f"Tipo WKB no soportado: {wkb_type}")

    return polys


def points_in_ring(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Algoritmo ray-casting vectorizado en NumPy para determinar puntos dentro de un anillo poligonal."""
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    cond = (y1[None, :] > py[:, None]) != (y2[None, :] > py[:, None])
    dy = np.where(y2 != y1, y2 - y1, 1.0)
    x_intersect = (x2[None, :] - x1[None, :]) * (py[:, None] - y1[None, :]) / dy[None, :] + x1[None, :]
    crossings = cond & (px[:, None] < x_intersect)
    return (np.sum(crossings, axis=1) % 2) == 1


def extract_training_data(
    scpx_path: str,
    mosaic_mm: np.memmap,
    meta: dict,
    output_tsv: str,
) -> pd.DataFrame:
    """Extrae los píxeles de entrenamiento definidos en AfterProy2.scpx y guarda el TSV."""
    t0 = time.time()
    print(f"[1/2] Procesando archivo de entrenamiento: {scpx_path} ...")

    with zipfile.ZipFile(scpx_path, "r") as z:
        gpkg_bytes = z.read("geometry.gpkg")

    # Leer polígonos ROI desde geometry.gpkg en memoria
    conn = sqlite3.connect(":memory:")
    conn.deserialize(gpkg_bytes)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'gpkg_%' AND name NOT LIKE 'rtree_%' AND name NOT LIKE 'sqlite_%'"
    )
    roi_table = cur.fetchone()[0]
    cur.execute(f"SELECT fid, roi_id, macroclass_id, class_id, geom FROM {roi_table}")
    roi_rows = cur.fetchall()
    conn.close()
    print(f"      Polígonos ROI leídos desde '{roi_table}': {len(roi_rows)}")

    x0, y0 = meta["x0"], meta["y0"]
    dx, dy = meta["dx"], meta["dy"]
    width, height = meta["width"], meta["height"]
    nodata = meta["nodata"]

    rows_list, cols_list, mc_id_list = [], [], []

    for fid, roi_id, m_id, c_id, geom_bytes in roi_rows:
        polys = parse_gpkg_geom(geom_bytes)
        for rings in polys:
            ext = rings[0]
            c_min = max(0, int(np.floor((ext[:, 0].min() - x0) / dx)))
            c_max = min(width - 1, int(np.ceil((ext[:, 0].max() - x0) / dx)))
            r_min = max(0, int(np.floor((y0 - ext[:, 1].max()) / dy)))
            r_max = min(height - 1, int(np.ceil((y0 - ext[:, 1].min()) / dy)))

            if c_min > c_max or r_min > r_max:
                continue

            cols_grid = np.arange(c_min, c_max + 1, dtype=np.int32)
            rows_grid = np.arange(r_min, r_max + 1, dtype=np.int32)
            cc, rr = np.meshgrid(cols_grid, rows_grid)
            cc_flat = cc.ravel()
            rr_flat = rr.ravel()

            # Coordenadas del centro de cada píxel de 20m
            px = x0 + (cc_flat + 0.5) * dx
            py = y0 - (rr_flat + 0.5) * dy

            inside = points_in_ring(px, py, ext)
            for hole in rings[1:]:
                inside &= ~points_in_ring(px, py, hole)

            if np.any(inside):
                rows_list.append(rr_flat[inside])
                cols_list.append(cc_flat[inside])
                mc_id_list.append(np.full(int(np.sum(inside)), m_id, dtype=np.int32))

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_mc = np.concatenate(mc_id_list)

    # Eliminar posibles píxeles duplicados si dos ROIs se solapan
    rc_pairs = np.column_stack([all_rows, all_cols])
    _, unique_idx = np.unique(rc_pairs, axis=0, return_index=True)
    unique_idx.sort()
    all_rows = all_rows[unique_idx]
    all_cols = all_cols[unique_idx]
    all_mc = all_mc[unique_idx]

    # Extraer valores de las 10 bandas desde el mosaico
    pixel_values = mosaic_mm[all_rows, all_cols].astype(np.int32)

    # Filtrar cualquier píxel NoData (todas las bandas en 0)
    valid_mask = ~np.all(pixel_values == int(nodata), axis=-1)
    all_rows = all_rows[valid_mask]
    all_cols = all_cols[valid_mask]
    all_mc = all_mc[valid_mask]
    pixel_values = pixel_values[valid_mask]

    # Calcular coordenadas proyectadas (centro del píxel en EPSG:7856)
    x_coords = np.round(x0 + (all_cols + 0.5) * dx, 4)
    y_coords = np.round(y0 - (all_rows + 0.5) * dy, 4)

    df_dict = {
        "row": all_rows,
        "col": all_cols,
        "x": x_coords,
        "y": y_coords,
    }
    for i, bname in enumerate(BAND_NAMES):
        df_dict[bname] = pixel_values[:, i]

    df_dict["macroclass_id"] = all_mc
    df_train = guardar_tsv_entrenamiento(pd.DataFrame(df_dict), output_tsv)
    print(f"      Tiempo de extracción y tabulación: {time.time() - t0:.2f} s")
    return df_train


def extract_full_mosaic(
    mosaic_mm: np.memmap,
    meta: dict,
    output_tsv: str,
    chunk_rows: int = 250,
) -> int:
    """Extrae todos los píxeles válidos (sin NoData=0) del Mosaico Completo por bloques de filas en formato TSV."""
    t0 = time.time()
    print(f"\n[2/2] Procesando Mosaico Completo hacia: {output_tsv} ...")

    width, height = meta["width"], meta["height"]
    x0, y0 = meta["x0"], meta["y0"]
    dx, dy = meta["dx"], meta["dy"]
    nodata = int(meta["nodata"])

    # Precalcular coordenadas X para todas las columnas (centro del píxel)
    col_indices_all = np.arange(width, dtype=np.int32)
    x_all = np.round(x0 + (col_indices_all + 0.5) * dx, 4)

    header_cols = ["row", "col", "x", "y"] + BAND_NAMES
    total_valid = 0

    # Escribir encabezado y procesar por bloques para mantener bajo uso de RAM y alta velocidad
    with open(output_tsv, "w", encoding="utf-8", newline="\n", buffering=16 * 1024 * 1024) as f_out:
        f_out.write("\t".join(header_cols) + "\n")

        for r_start in range(0, height, chunk_rows):
            r_end = min(height, r_start + chunk_rows)
            block = mosaic_mm[r_start:r_end]  # shape: (n_rows, width, 10)

            # Píxeles válidos: aquellos donde no todas las bandas son 0 (NoData)
            valid_mask = np.any(block != nodata, axis=-1)
            if not np.any(valid_mask):
                continue

            local_r, cols = np.nonzero(valid_mask)
            rows = (local_r + r_start).astype(np.int32)
            cols = cols.astype(np.int32)
            bands_vals = block[local_r, cols].astype(np.int32)

            x_vals = x_all[cols]
            y_vals = np.round(y0 - (rows + 0.5) * dy, 4)

            df_chunk = pd.DataFrame({
                "row": rows,
                "col": cols,
                "x": x_vals,
                "y": y_vals,
                "B01": bands_vals[:, 0],
                "B02": bands_vals[:, 1],
                "B03": bands_vals[:, 2],
                "B04": bands_vals[:, 3],
                "B05": bands_vals[:, 4],
                "B06": bands_vals[:, 5],
                "B07": bands_vals[:, 6],
                "B08": bands_vals[:, 7],
                "B11": bands_vals[:, 8],
                "B12": bands_vals[:, 9],
            })
            df_chunk.to_csv(f_out, sep="\t", header=False, index=False, lineterminator="\n")

            total_valid += len(rows)
            if (r_end % 1000 == 0) or (r_end == height):
                elapsed = time.time() - t0
                pct = (r_end / height) * 100
                print(f"      Progreso: {r_end}/{height} filas ({pct:.1f}%) | "
                      f"Píxeles válidos acumulados: {total_valid:,} | {elapsed:.1f} s")

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_tsv) / (1024 * 1024)
    print(f"      Archivo generado: {output_tsv} ({size_mb:,.1f} MB)")
    print(f"      Total píxeles válidos exportados: {total_valid:,} ({elapsed:.2f} s)")
    return total_valid


def main():
    parser = argparse.ArgumentParser(description="Genera las tabulaciones de la Entrega 3.")
    parser.add_argument(
        "--desde-tsv",
        action="store_true",
        help="Actualiza AfterProy2_Training_Tabulado.tsv usando píxeles existentes.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_tsv_path = os.path.join(base_dir, "AfterProy2_Training_Tabulado.tsv")
    if args.desde_tsv:
        guardar_tsv_entrenamiento(pd.read_csv(train_tsv_path, sep="\t"), train_tsv_path)
        return

    tif_path = os.path.join(base_dir, "Mosaico_Completo.tif")
    scpx_path = os.path.join(base_dir, "AfterProy2.scpx")
    mosaic_tsv_path = os.path.join(base_dir, "Mosaico_Completo_Tabulado.tsv")

    meta = read_geotiff_metadata(tif_path)
    print("=== Metadatos de Mosaico_Completo.tif ===")
    print(f"Dimensiones: {meta['width']} x {meta['height']} ({meta['width'] * meta['height']:,} píxeles totales)")
    print(f"Bandas: {meta['num_bands']} ({', '.join(BAND_NAMES)})")
    print(f"Resolución: {meta['dx']}m x {meta['dy']}m | Origen (x0, y0): ({meta['x0']}, {meta['y0']})")
    print("=========================================\n")

    mosaic_mm = np.memmap(
        tif_path,
        dtype=np.float32,
        mode="r",
        offset=meta["data_offset"],
        shape=(meta["height"], meta["width"], meta["num_bands"]),
    )

    extract_training_data(scpx_path, mosaic_mm, meta, train_tsv_path)
    extract_full_mosaic(mosaic_mm, meta, mosaic_tsv_path)


if __name__ == "__main__":
    main()
