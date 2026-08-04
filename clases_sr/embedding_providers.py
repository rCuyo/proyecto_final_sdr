"""
Capa de abstracción para la generación de embeddings de la consulta del usuario.

Encapsula los DOS flujos que ya existían dispersos en el proyecto del docente y en
nuestro propio trabajo, para que `main.py` pueda elegir uno u otro sin `if/else`
repartidos por el endpoint:

  - RemoteEmbeddingProvider: el flujo ORIGINAL del docente (huggingface_hub.InferenceClient
    contra la Inference API de Hugging Face, modelo BAAI/bge-m3). Código portado
    literalmente del patrón de `consulta1.py` / `pruebatoken.py`, sin reinventarlo.
  - LocalEmbeddingProvider: el flujo que ya usa `main.py` hoy (SentenceTransformer
    cargado en memoria, nuestro modelo Fine-Tuned). Es un envoltorio delgado: no vuelve
    a cargar el modelo, recibe la instancia ya creada por `main.py`.

Ninguna de las dos clases decide POR SÍ SOLA cuál usar -esa decisión sigue siendo de
`main.py`, en base al campo `modo` de la request-. Este módulo solo ofrece una interfaz
común (`generar_embedding`) para que ese punto de decisión sea un único diccionario, no
condicionales dispersos.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProvider(ABC):
    """Interfaz común: dado un texto, devuelve su embedding como lista de floats."""

    @abstractmethod
    def generar_embedding(self, texto: str) -> List[float]:
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Modo LOCAL: envuelve un SentenceTransformer ya cargado (nuestro modelo Fine-Tuned,
    `models/bge-m3-finetuned`, cargado una sola vez por `main.py` al iniciar).

    No instancia su propio modelo -evita cargarlo dos veces en memoria-; recibe la
    misma instancia que `main.py` ya usaba antes de este cambio.
    """

    def __init__(self, modelo) -> None:
        self._modelo = modelo

    def generar_embedding(self, texto: str) -> List[float]:
        # Mismo patrón que main.py usaba antes de esta capa (idéntico, no reescrito).
        return self._modelo.encode(texto, normalize_embeddings=True).tolist()


class RemoteEmbeddingProvider(EmbeddingProvider):
    """
    Modo ONLINE: reutiliza el flujo original del docente -InferenceClient de
    `huggingface_hub` contra la Inference API de Hugging Face-, tal como está en
    `clases_sr/consulta1.py` y `clases_sr/pruebatoken.py`.

    El token NO se valida acá (construir `InferenceClient` no hace ninguna llamada de
    red): si `api_key` es None o inválido, el error recién aparece cuando se llama a
    `generar_embedding`, y ese error ya lo captura el mismo `try/except` que main.py
    tiene desde antes. Esto permite que la app arranque igual aunque no haya
    `HF_TOKEN` configurado, siempre que nadie elija el modo online.
    """

    MODEL = "BAAI/bge-m3"

    def __init__(self, api_key: Optional[str]) -> None:
        from huggingface_hub import InferenceClient

        # Mismo patrón que consulta1.py / pruebatoken.py (provider="hf-inference",
        # api_key=...), sin reinventar la llamada del docente.
        self._client = InferenceClient(provider="hf-inference", api_key=api_key)

    def generar_embedding(self, texto: str) -> List[float]:
        embedding = self._client.feature_extraction(texto, model=self.MODEL)
        # feature_extraction devuelve un numpy.ndarray en la mayoría de las versiones
        # de huggingface_hub; se normaliza a list[float] para que sea intercambiable
        # con la salida de LocalEmbeddingProvider (ChromaDB espera una lista/array).
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
