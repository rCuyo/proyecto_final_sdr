"""
Preparación del dataset "The Movies Dataset" (Kaggle) para el Recomendador de Películas.

Esta es la ETAPA 1 del proyecto: solo descarga, inspecciona y limpia el dataset.
Todavía NO se generan embeddings, NO se construye ChromaDB y NO se toca FastAPI.

Requiere:
    pip install kagglehub pandas
    (credenciales de Kaggle configuradas: ~/.kaggle/kaggle.json
     o variables de entorno KAGGLE_USERNAME / KAGGLE_KEY)
"""

import ast
from pathlib import Path
from typing import Dict

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter  # noqa: F401  (referencia, ver leer_datasets)

# ===============================
# CONFIGURACIÓN
# ===============================

# Identificador del dataset en Kaggle (owner/nombre-dataset)
DATASET_HANDLE = "rounakbanik/the-movies-dataset"

# Únicamente estos archivos son útiles para el recomendador basado en contenido:
#   - movies_metadata.csv: título, sinopsis, género -> texto principal para embeddings
#   - keywords.csv:        palabras clave temáticas -> enriquecen la descripción semántica
#   - credits.csv:         reparto/director -> permite búsquedas por actor o director
# Se descartan ratings*.csv (filtrado colaborativo, no aplica) y links*.csv (solo IDs externos).
ARCHIVOS_REQUERIDOS = ["movies_metadata.csv", "keywords.csv", "credits.csv"]

# Carpeta base del proyecto reutilizado (clases_sr/) y carpeta de salida
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PROCESADA_DIR = DATA_DIR / "processed"

# Columnas que, a priori, aportan poco valor para un recomendador semántico
# (no se eliminan del dataset limpio, solo se reportan como candidatas a descartar más adelante)
COLUMNAS_IRRELEVANTES = {
    "movies_metadata.csv": [
        "adult", "homepage", "poster_path", "video",
        "belongs_to_collection", "original_title", "status",
    ],
    "keywords.csv": [],
    "credits.csv": [],
}


# ===============================
# 1. DESCARGA DEL DATASET
# ===============================
def descargar_dataset() -> Path:
    """
    Descarga (o reutiliza la copia en caché) de cada archivo requerido de
    "The Movies Dataset" usando kagglehub, y devuelve la carpeta local donde
    quedaron los archivos.

    kagglehub descarga solo el/los archivo(s) indicado(s) en `path`, evitando
    traer archivos que no vamos a usar (ratings.csv, links.csv, etc.).
    """
    print("=" * 60)
    print("DESCARGANDO DATASET DESDE KAGGLE")
    print("=" * 60)

    carpeta_dataset = None

    for archivo in ARCHIVOS_REQUERIDOS:
        print(f"Descargando '{archivo}'...")
        try:
            ruta_descarga = kagglehub.dataset_download(DATASET_HANDLE, path=archivo)
        except Exception as error:
            raise RuntimeError(
                "No se pudo descargar el dataset desde Kaggle. "
                "Verifica tus credenciales (~/.kaggle/kaggle.json o "
                "las variables de entorno KAGGLE_USERNAME / KAGGLE_KEY)."
            ) from error

        ruta_descarga = Path(ruta_descarga)
        # Si kagglehub devuelve la carpeta contenedora en vez del archivo puntual,
        # nos quedamos con la carpeta; si devuelve el archivo, usamos su padre.
        carpeta_dataset = ruta_descarga if ruta_descarga.is_dir() else ruta_descarga.parent

    print(f"Dataset disponible en: {carpeta_dataset}")
    return carpeta_dataset


# ===============================
# 2. LECTURA DE LOS ARCHIVOS NECESARIOS
# ===============================
def leer_datasets(carpeta_dataset: Path) -> Dict[str, pd.DataFrame]:
    """
    Lee únicamente los 3 archivos que usará el recomendador y los devuelve
    en un diccionario {nombre_archivo: DataFrame}.
    """
    print("\n" + "=" * 60)
    print("LEYENDO ARCHIVOS NECESARIOS")
    print("=" * 60)

    dataframes = {}

    for nombre_archivo in ARCHIVOS_REQUERIDOS:
        ruta_archivo = carpeta_dataset / nombre_archivo

        if not ruta_archivo.exists():
            raise FileNotFoundError(
                f"No se encontró '{nombre_archivo}' en {carpeta_dataset}"
            )

        # low_memory=False evita el DtypeWarning conocido de movies_metadata.csv,
        # que tiene algunas filas corruptas con tipos mixtos en la columna 'id'.
        # compression='zip' si corresponde: la descarga por archivo individual de
        # kagglehub entrega un ZIP con el mismo nombre ".csv" (el contenido real
        # está comprimido), no el CSV plano.
        compresion = "zip" if _es_zip(ruta_archivo) else None
        df = pd.read_csv(ruta_archivo, low_memory=False, compression=compresion)
        dataframes[nombre_archivo] = df
        print(f"'{nombre_archivo}' leído: {len(df)} filas, {len(df.columns)} columnas")

    return dataframes


def _es_zip(ruta: Path) -> bool:
    """Detecta si un archivo está en realidad comprimido en ZIP (firma 'PK') pese a su extensión."""
    with open(ruta, "rb") as archivo:
        return archivo.read(4) == b"PK\x03\x04"


# ===============================
# 3. RESUMEN DEL DATAFRAME
# ===============================
def mostrar_resumen(df: pd.DataFrame, nombre: str) -> None:
    """Muestra cantidad de registros, columnas, tipos de datos y primeras filas."""
    print("\n" + "-" * 60)
    print(f"RESUMEN: {nombre}")
    print("-" * 60)
    print(f"Registros: {len(df)}")
    print(f"Columnas ({len(df.columns)}): {list(df.columns)}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nPrimeros registros:")
    print(df.head(3))


# ===============================
# 4. DETECCIÓN DE PROBLEMAS COMUNES
# ===============================
def detectar_problemas(df: pd.DataFrame, nombre: str) -> None:
    """
    Detecta y reporta problemas típicos de este dataset:
    valores nulos, filas duplicadas, IDs duplicados y columnas
    candidatas a ser irrelevantes para el recomendador.
    """
    print("\n" + "-" * 60)
    print(f"DIAGNÓSTICO: {nombre}")
    print("-" * 60)

    # --- Valores nulos ---
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if nulos.empty:
        print("Valores nulos: ninguno")
    else:
        print("Valores nulos por columna:")
        for columna, cantidad in nulos.items():
            porcentaje = cantidad / len(df) * 100
            print(f"  - {columna}: {cantidad} ({porcentaje:.1f}%)")

    # --- Filas totalmente duplicadas ---
    duplicados_totales = df.duplicated().sum()
    print(f"Filas duplicadas: {duplicados_totales}")

    # --- IDs duplicados / inválidos (columna 'id') ---
    if "id" in df.columns:
        ids_no_numericos = pd.to_numeric(df["id"], errors="coerce").isna().sum()
        print(f"IDs no numéricos / corruptos: {ids_no_numericos}")

        ids_duplicados = df["id"].duplicated().sum()
        print(f"IDs duplicados: {ids_duplicados}")

    # --- Columnas candidatas a irrelevantes ---
    candidatas = COLUMNAS_IRRELEVANTES.get(nombre, [])
    candidatas_presentes = [c for c in candidatas if c in df.columns]
    if candidatas_presentes:
        print(f"Columnas candidatas a irrelevantes: {candidatas_presentes}")


# ===============================
# Utilidades de limpieza
# ===============================
def _limpiar_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'id' a numérico y descarta filas con id inválido o duplicado."""
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates()  # filas 100% duplicadas
    df = df.drop_duplicates(subset=["id"], keep="first")  # mismo id repetido
    return df


def _parsear_lista_dict(valor) -> list:
    """
    Convierte columnas tipo string "[{'id': 1, 'name': 'Action'}, ...]"
    (formato habitual de genres/keywords/cast/crew en este dataset) en una
    lista real de diccionarios. Devuelve [] si el valor es nulo o inválido.
    """
    if pd.isna(valor):
        return []
    try:
        resultado = ast.literal_eval(valor)
        return resultado if isinstance(resultado, list) else []
    except (ValueError, SyntaxError):
        return []


def _nombres_de(lista_dicts: list, limite: int = None) -> str:
    """Une los campos 'name' de una lista de diccionarios en un string separado por comas."""
    nombres = [d.get("name", "") for d in lista_dicts if isinstance(d, dict) and d.get("name")]
    if limite:
        nombres = nombres[:limite]
    return ", ".join(nombres)


def _extraer_director(lista_crew: list) -> str:
    """Busca en la lista de 'crew' a la persona cuyo job sea 'Director'."""
    for persona in lista_crew:
        if isinstance(persona, dict) and persona.get("job") == "Director":
            return persona.get("name", "")
    return ""


# ===============================
# 7. GUARDAR DATASET LIMPIO
# ===============================
def guardar_dataset_limpio(dataframes: Dict[str, pd.DataFrame]) -> Path:
    """
    Limpia estructuralmente los 3 datasets (ids inválidos/duplicados, filas
    duplicadas), los combina en un único DataFrame por 'id' y agrega columnas
    derivadas ('categoria', 'descripcion', 'director', 'reparto') con la misma
    semántica que 'peliculas_200.csv', sin eliminar ninguna columna original.
    """
    print("\n" + "=" * 60)
    print("LIMPIANDO Y GUARDANDO DATASET")
    print("=" * 60)

    movies = _limpiar_ids(dataframes["movies_metadata.csv"])
    keywords = _limpiar_ids(dataframes["keywords.csv"])
    credits = _limpiar_ids(dataframes["credits.csv"])

    # --- Combinar por id, conservando todas las películas aunque falten keywords/credits ---
    df = movies.merge(keywords[["id", "keywords"]], on="id", how="left")
    df = df.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    # --- Parsear columnas JSON-like a texto legible ---
    df["categoria"] = df["genres"].apply(_parsear_lista_dict).apply(_nombres_de)
    keywords_texto = df["keywords"].apply(_parsear_lista_dict).apply(_nombres_de)
    df["reparto"] = df["cast"].apply(_parsear_lista_dict).apply(lambda lista: _nombres_de(lista, limite=5))
    df["director"] = df["crew"].apply(_parsear_lista_dict).apply(_extraer_director)

    # --- Descripción semántica: sinopsis + palabras clave (si existen) ---
    overview = df["overview"].fillna("")
    df["descripcion"] = [
        f"{ov.strip()} Palabras clave: {kw}".strip() if kw else ov.strip()
        for ov, kw in zip(overview, keywords_texto)
    ]

    # Renombrar 'title' a 'titulo' para alinear con la nomenclatura del proyecto
    df = df.rename(columns={"title": "titulo"})

    # --- Guardar ---
    DATA_PROCESADA_DIR.mkdir(parents=True, exist_ok=True)
    ruta_salida = DATA_PROCESADA_DIR / "peliculas_dataset_limpio.csv"
    df.to_csv(ruta_salida, index=False)

    print(f"Registros finales: {len(df)}")
    print(f"Guardado en: {ruta_salida}")
    return ruta_salida


# ===============================
# ORQUESTACIÓN
# ===============================
def main() -> None:
    carpeta_dataset = descargar_dataset()
    dataframes = leer_datasets(carpeta_dataset)

    for nombre, df in dataframes.items():
        mostrar_resumen(df, nombre)
        detectar_problemas(df, nombre)

    guardar_dataset_limpio(dataframes)

    print("\n" + "=" * 60)
    print("ETAPA 1 COMPLETADA: dataset descargado, revisado y preparado.")
    print("(Todavía sin embeddings, sin ChromaDB, sin cambios en FastAPI)")
    print("=" * 60)


if __name__ == "__main__":
    main()
