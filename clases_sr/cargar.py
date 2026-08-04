"""
Construye la base vectorial ChromaDB ("peliculas") usando el modelo
Fine-Tuned (models/bge-m3-finetuned) en vez del BAAI/bge-m3 original.

Reutiliza la lógica del cargar.py original del docente (borrar base anterior,
crear cliente persistente, colección "peliculas", insertar documentos con
metadata) y la carga/preparación de datos de la etapa 2
(`generar_dataset_entrenamiento.cargar_dataset` + `crear_columna_texto`), para
que el texto que se embebe sea EXACTAMENTE el mismo formato usado durante el
fine-tuning. Ya no se usa la API remota de Hugging Face (`InferenceClient`):
el modelo fine-tuned es un checkpoint local, así que la inferencia también es
local, con el mismo patrón que `ubica_modelo.py` ya usaba en este proyecto.

No modifica main.py, ni el pipeline de entrenamiento, ni el dataset.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

# generar_dataset_entrenamiento.py vive en scripts/, un nivel arriba de clases_sr/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generar_dataset_entrenamiento import cargar_dataset, crear_columna_texto  # noqa: E402

# ===============================
# CONFIGURACIÓN
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent  # raíz del proyecto (clases_sr/..)

RUTA_MODELO = BASE_DIR / "models" / "bge-m3-finetuned"
RUTA_DB = Path(__file__).resolve().parent / "chroma_db"
NOMBRE_COLECCION = "peliculas"

DIMENSION_ESPERADA = 1024  # dimensión de embedding de bge-m3
BATCH_SIZE_ENCODING = 64  # lote para SentenceTransformer.encode
TAM_LOTE_INSERCION = 1000  # ChromaDB limita a ~5461 documentos por add(); se deja margen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cargar")


# ===============================
# Modelo
# ===============================
def cargar_modelo(ruta_modelo: Path) -> SentenceTransformer:
    """Carga el modelo Fine-Tuned local (auto-detecta CUDA si está disponible)."""
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo Fine-Tuned en '{ruta_modelo}'. "
            "Ejecuta primero scripts/finetuning.py."
        )
    logger.info("Cargando modelo Fine-Tuned desde: %s", ruta_modelo)
    modelo = SentenceTransformer(str(ruta_modelo))
    logger.info("Modelo cargado (dimensión de embedding: %d).", modelo.get_sentence_embedding_dimension())
    return modelo


# ===============================
# Datos
# ===============================
def cargar_peliculas() -> pd.DataFrame:
    """
    Carga el dataset real (etapa 1) y construye la columna 'texto' (etapa 2),
    reutilizando exactamente las mismas funciones usadas para preparar los
    datos de entrenamiento, para que el texto indexado coincida con el
    formato con el que se hizo el fine-tuning.
    """
    df = cargar_dataset()
    df = crear_columna_texto(df)
    logger.info("Películas cargadas para indexar: %d", len(df))
    return df


# ===============================
# ChromaDB
# ===============================
def eliminar_base_anterior(ruta_db: Path) -> None:
    """Elimina completamente la base ChromaDB anterior (no se reutiliza ningún embedding viejo)."""
    if ruta_db.exists():
        logger.info("Eliminando base ChromaDB anterior: %s", ruta_db)
        shutil.rmtree(ruta_db)
    else:
        logger.info("No había una base ChromaDB previa en: %s", ruta_db)


def crear_coleccion(ruta_db: Path, nombre: str) -> Tuple[chromadb.ClientAPI, Collection]:
    """Crea un cliente persistente nuevo y la colección (mismo nombre de siempre, para no romper la API)."""
    logger.info("Creando nueva base ChromaDB en: %s", ruta_db)
    cliente = chromadb.PersistentClient(path=str(ruta_db))
    coleccion = cliente.get_or_create_collection(name=nombre)
    return cliente, coleccion


# ===============================
# Embeddings
# ===============================
def generar_embeddings(modelo: SentenceTransformer, textos: List[str], batch_size: int) -> np.ndarray:
    """Genera TODOS los embeddings con el modelo Fine-Tuned, en lotes, normalizados (coseno)."""
    logger.info("Generando embeddings para %d películas (batch_size=%d)...", len(textos), batch_size)
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info("Embeddings generados: %s", embeddings.shape)
    return embeddings


def insertar_documentos(
    coleccion: Collection, df: pd.DataFrame, embeddings: np.ndarray, tam_lote: int
) -> None:
    """
    Inserta documentos + embeddings + metadata en lotes (ChromaDB limita la
    cantidad de documentos por `add()`). La metadata guarda la sinopsis
    original ('overview'), no el texto estructurado completo que se embebe,
    para que /buscar siga devolviendo una descripción limpia.
    """
    total = len(df)
    ids = df["id"].astype(str).tolist()
    documentos = df["texto"].tolist()
    titulos = df["titulo"].fillna("").astype(str)
    categorias = df["categoria"].fillna("").astype(str)
    descripciones = df["overview"].fillna("").astype(str)

    for inicio in range(0, total, tam_lote):
        fin = min(inicio + tam_lote, total)
        metadatas: List[Dict[str, Any]] = [
            {
                "titulo": titulos.iloc[i],
                "categoria": categorias.iloc[i],
                "descripcion": descripciones.iloc[i],
            }
            for i in range(inicio, fin)
        ]
        coleccion.add(
            ids=ids[inicio:fin],
            documents=documentos[inicio:fin],
            embeddings=embeddings[inicio:fin],
            metadatas=metadatas,
        )
        logger.info("Insertadas %d/%d películas...", fin, total)


# ===============================
# Verificación (Fase 4)
# ===============================
def verificar_base(
    coleccion: Collection, df: pd.DataFrame, embeddings: np.ndarray, dimension_esperada: int
) -> Dict[str, Any]:
    """
    Verifica que número de documentos, embeddings e IDs coincidan, que no
    haya IDs duplicados ni embeddings vacíos, y que la dimensión sea la
    esperada. Devuelve un resumen y lo muestra por consola.
    """
    n_documentos_df = len(df)
    n_embeddings = embeddings.shape[0]
    n_en_coleccion = coleccion.count()

    ids_duplicados = int(df["id"].astype(str).duplicated().sum())
    embeddings_vacios = int(np.sum(~np.any(embeddings, axis=1))) if n_embeddings else 0
    dimension_real = embeddings.shape[1] if embeddings.ndim == 2 else 0

    resumen = {
        "documentos_dataframe": n_documentos_df,
        "embeddings_generados": n_embeddings,
        "documentos_en_coleccion": n_en_coleccion,
        "ids_duplicados": ids_duplicados,
        "embeddings_vacios": embeddings_vacios,
        "dimension_embedding": dimension_real,
        "dimension_esperada": dimension_esperada,
        "todo_coincide": (
            n_documentos_df == n_embeddings == n_en_coleccion
            and ids_duplicados == 0
            and embeddings_vacios == 0
            and dimension_real == dimension_esperada
        ),
    }

    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE LA BASE VECTORIAL")
    print("=" * 60)
    print(f"Documentos en el dataset:      {resumen['documentos_dataframe']}")
    print(f"Embeddings generados:          {resumen['embeddings_generados']}")
    print(f"Documentos en la colección:    {resumen['documentos_en_coleccion']}")
    print(f"IDs duplicados:                {resumen['ids_duplicados']}")
    print(f"Embeddings vacíos:             {resumen['embeddings_vacios']}")
    print(f"Dimensión de embedding:        {resumen['dimension_embedding']} (esperada: {dimension_esperada})")
    print(f"Todo coincide:                 {'SÍ' if resumen['todo_coincide'] else 'NO'}")
    print("=" * 60)

    if not resumen["todo_coincide"]:
        logger.warning("La verificación de la base vectorial encontró inconsistencias: %s", resumen)

    return resumen


# ===============================
# ORQUESTACIÓN
# ===============================
def main() -> None:
    try:
        modelo = cargar_modelo(RUTA_MODELO)
        df = cargar_peliculas()

        eliminar_base_anterior(RUTA_DB)
        _, coleccion = crear_coleccion(RUTA_DB, NOMBRE_COLECCION)

        embeddings = generar_embeddings(modelo, df["texto"].tolist(), BATCH_SIZE_ENCODING)
        insertar_documentos(coleccion, df, embeddings, TAM_LOTE_INSERCION)

        verificar_base(coleccion, df, embeddings, DIMENSION_ESPERADA)

        print(f"\nBase vectorial '{NOMBRE_COLECCION}' creada correctamente con el modelo Fine-Tuned.")
    except Exception:
        logger.exception("Falló la construcción de la base vectorial.")
        raise


if __name__ == "__main__":
    main()
