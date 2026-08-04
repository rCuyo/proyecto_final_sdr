import os

# Debe fijarse ANTES de importar sentence_transformers/transformers: el entorno tiene
# un TensorFlow instalado (ajeno a este proyecto) incompatible con Keras 3, lo que
# rompe el import si TF queda habilitado. Mismo workaround ya usado en
# scripts/finetuning.py; este proyecto solo usa el backend de PyTorch.
os.environ.setdefault("USE_TF", "0")

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

from embedding_providers import LocalEmbeddingProvider, RemoteEmbeddingProvider

# ==========================
# Configuración
# ==========================
# main.py vive en clases_sr/; el modelo Fine-Tuned está en la raíz del proyecto.
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "models" / "bge-m3-finetuned"

# Colección "peliculas_original" ya generada por scripts/verificar_migracion.py:
# las mismas 44 471 películas embebidas con el modelo BAAI/bge-m3 ORIGINAL (sin
# fine-tuning). El modo online la usa en vez de "peliculas" porque esta última fue
# embebida con nuestro modelo Fine-Tuned -mezclar un embedding remoto (modelo
# original) contra esa colección compararía dos espacios de embeddings distintos-.
# No se genera nada nuevo: es el mismo artefacto que ya existe en disco.
RUTA_DB_ORIGINAL = BASE_DIR / "reports" / "_chroma_comparacion_original"
NOMBRE_COLECCION_ORIGINAL = "peliculas_original"

app = FastAPI(
    title="API Recomendador de Películas",
    version="1.0"
)

# ==========================
# CORS
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Inicialización
# ==========================

modelo = SentenceTransformer(str(RUTA_MODELO))

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    "peliculas"
)

# Segundo cliente ChromaDB, apuntando a la colección del modelo original (ver
# comentario junto a RUTA_DB_ORIGINAL). Es un directorio distinto en disco, pero
# sigue siendo un único proceso de FastAPI, un único endpoint /buscar.
chroma_client_original = chromadb.PersistentClient(
    path=str(RUTA_DB_ORIGINAL)
)

collection_original = chroma_client_original.get_collection(
    NOMBRE_COLECCION_ORIGINAL
)

# ==========================
# Proveedores de embedding (Modo Online / Modo Local)
# ==========================
# Único punto de decisión entre ambos modos: qué EmbeddingProvider y qué colección de
# ChromaDB usar. buscar_peliculas() solo indexa estos dos diccionarios por `data.modo`
# -no hay más if/else de modo repartidos en el resto del archivo-.
PROVEEDORES = {
    "local": LocalEmbeddingProvider(modelo),
    "online": RemoteEmbeddingProvider(api_key=os.environ.get("HF_TOKEN")),
}

COLECCIONES = {
    "local": collection,
    "online": collection_original,
}

# ==========================
# Modelos de entrada
# ==========================

class ConsultaRequest(BaseModel):
    consulta: str
    n_resultados: int = 5
    modo: Literal["local", "online"] = "local"

# ==========================
# Endpoint
# ==========================
@app.get("/")
def inicio():

    return FileResponse(
        "static/index.html"
    )
@app.post("/buscar")
def buscar_peliculas(data: ConsultaRequest):

    try:

        proveedor = PROVEEDORES[data.modo]
        coleccion_activa = COLECCIONES[data.modo]

        embedding = proveedor.generar_embedding(data.consulta)

        resultado = coleccion_activa.query(
            query_embeddings=[embedding],
            n_results=data.n_resultados
        )

        peliculas = []

        for meta in resultado["metadatas"][0]:

            peliculas.append({
                "titulo": meta["titulo"],
                "categoria": meta["categoria"],
                # La colección del modo online ("peliculas_original") no guarda
                # "descripcion" en su metadata (ver comentario junto a RUTA_DB_ORIGINAL);
                # .get(...) evita un KeyError sin afectar al modo local, que sí la tiene.
                "descripcion": meta.get("descripcion", "")
            })

        return {
            "consulta": data.consulta,
            "cantidad": len(peliculas),
            "resultados": peliculas
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )