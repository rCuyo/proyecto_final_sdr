"""
Construcción del dataset de entrenamiento para el Fine-Tuning de BAAI/bge-m3.

Esta es la ETAPA 2 del proyecto: a partir de `data/processed/peliculas_dataset_limpio.csv`
(generado por `preparar_dataset.py`) se construyen pares positivos, pares negativos y
tripletas (anchor, positive, negative) listas para entrenar un modelo de embeddings
con triplet loss.

Todavía NO se generan embeddings, NO se construye ChromaDB, NO se toca FastAPI y
NO se entrena ningún modelo. Solo se prepara el dataset de entrenamiento.

Requiere:
    pip install pandas numpy scikit-learn
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List

import ast
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# CONFIGURACIÓN
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATASET_LIMPIO = BASE_DIR / "data" / "processed" / "peliculas_dataset_limpio.csv"
DATA_TRAINING_DIR = BASE_DIR / "data" / "training"

# Columnas mínimas que debe traer el CSV de la etapa 1
COLUMNAS_REQUERIDAS = ["id", "titulo", "categoria", "overview", "director", "reparto", "keywords"]

# --- Pesos del puntaje combinado para pares positivos ---
# Género y keywords pesan más porque son la señal temática más confiable;
# TF-IDF aporta la similitud de trama real; director/reparto son señales de apoyo.
PESO_GENERO = 0.30
PESO_KEYWORDS = 0.25
PESO_DIRECTOR = 0.15
PESO_REPARTO = 0.10
PESO_TFIDF = 0.20

# --- Umbrales ---
UMBRAL_TFIDF_MIN_POSITIVO = 0.05  # evita positivos que solo coinciden en género, sin relación de trama
UMBRAL_SCORE_POSITIVO = 0.35
UMBRAL_JACCARD_KEYWORDS_NEGATIVO = 0.05  # casi cero keywords en común
UMBRAL_TFIDF_MAX_NEGATIVO = 0.10

# --- Límites de rendimiento (evitan un cómputo O(n^2) sobre ~45k películas) ---
MAX_CANDIDATOS_POSITIVOS_POR_ANCLA = 200
MAX_MUESTREO_NEGATIVOS_POR_ANCLA = 80

N_POSITIVOS_POR_ANCLA = 2
N_NEGATIVOS_POR_ANCLA = 2
SEMILLA_ALEATORIA = 42

# Plantilla de texto: estructura fija para que el modelo aprenda un formato consistente
PLANTILLA_TEXTO = (
    "Título:\n{titulo}\n\n"
    "Categoría:\n{categoria}\n\n"
    "Director:\n{director}\n\n"
    "Reparto:\n{reparto}\n\n"
    "Palabras clave:\n{keywords}\n\n"
    "Descripción:\n{descripcion}"
)

VALOR_POR_DEFECTO = "No disponible"


# ===============================
# Estructuras de similitud (precomputadas una sola vez)
# ===============================
@dataclass
class EstructurasSimilitud:
    """Agrupa todo lo que se calcula una única vez y se reutiliza para positivos y negativos."""

    tfidf_matrix: csr_matrix
    generos_sets: List[FrozenSet[str]]
    keywords_sets: List[FrozenSet[str]]
    cast_sets: List[FrozenSet[str]]
    directores: List[str]
    indice_invertido_genero: Dict[str, List[int]]


# ===============================
# Utilidades de parseo (duplicadas intencionalmente)
# ===============================
# Nota: NO se importa `preparar_dataset.py` para no arrastrar su dependencia dura de
# kagglehub (esta etapa no descarga nada de Kaggle). Se reimplementa aquí el mismo
# parser de columnas tipo "[{'id': 1, 'name': 'Action'}, ...]" usado en la etapa 1.
def _parsear_lista_dict(valor) -> list:
    """Convierte un string con forma de lista de diccionarios en una lista real."""
    if pd.isna(valor):
        return []
    try:
        resultado = ast.literal_eval(valor)
        return resultado if isinstance(resultado, list) else []
    except (ValueError, SyntaxError):
        return []


def _nombres_de(lista_dicts: list) -> str:
    """Une los campos 'name' de una lista de diccionarios en un string separado por comas."""
    nombres = [d.get("name", "") for d in lista_dicts if isinstance(d, dict) and d.get("name")]
    return ", ".join(nombres)


def _a_conjunto(texto_separado_por_comas: str) -> FrozenSet[str]:
    """Convierte 'Action, Drama' -> frozenset({'action', 'drama'}) (normalizado en minúsculas)."""
    if not texto_separado_por_comas:
        return frozenset()
    return frozenset(t.strip().lower() for t in texto_separado_por_comas.split(",") if t.strip())


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Similitud de Jaccard entre dos conjuntos; 0.0 si alguno está vacío."""
    if not a or not b:
        return 0.0
    interseccion = len(a & b)
    union = len(a | b)
    return interseccion / union if union else 0.0


# ===============================
# 1. CARGA DEL DATASET
# ===============================
def cargar_dataset(ruta: Path = RUTA_DATASET_LIMPIO) -> pd.DataFrame:
    """
    Carga el dataset limpio generado en la etapa 1 y descarta filas sin
    título o sinopsis (no aportan valor para el entrenamiento de embeddings).
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró '{ruta}'. Ejecuta primero scripts/preparar_dataset.py."
        )

    df = pd.read_csv(ruta, low_memory=False)

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el dataset limpio: {faltantes}")

    df = df.dropna(subset=["titulo", "overview"])
    df = df[df["overview"].str.strip().astype(bool)]

    # Rellenar campos opcionales para que el resto del pipeline no tenga que manejar NaN
    for columna in ["categoria", "director", "reparto"]:
        df[columna] = df[columna].fillna("")

    df = df.reset_index(drop=True)
    print(f"Dataset cargado: {len(df)} películas con título y sinopsis válidos.")
    return df


# ===============================
# 2. COLUMNA "texto"
# ===============================
def crear_columna_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega:
      - 'keywords_limpio': texto de keywords parseado desde la columna JSON cruda.
      - 'texto': representación textual completa de la película, con el formato
        fijo pedido (Título / Categoría / Director / Reparto / Palabras clave / Descripción).

    No se elimina ninguna columna original.
    """
    df = df.copy()

    df["keywords_limpio"] = df["keywords"].apply(_parsear_lista_dict).apply(_nombres_de)

    def _construir_fila(fila: pd.Series) -> str:
        return PLANTILLA_TEXTO.format(
            titulo=fila["titulo"] or VALOR_POR_DEFECTO,
            categoria=fila["categoria"] or VALOR_POR_DEFECTO,
            director=fila["director"] or VALOR_POR_DEFECTO,
            reparto=fila["reparto"] or VALOR_POR_DEFECTO,
            keywords=fila["keywords_limpio"] or VALOR_POR_DEFECTO,
            descripcion=fila["overview"] or VALOR_POR_DEFECTO,
        )

    df["texto"] = df.apply(_construir_fila, axis=1)
    print("Columna 'texto' creada con el formato estructurado.")
    return df


# ===============================
# Construcción de estructuras de similitud (una sola vez)
# ===============================
def _construir_estructuras_similitud(df: pd.DataFrame) -> EstructurasSimilitud:
    """
    Precomputa todo lo necesario para puntuar similitud sin repetir trabajo:
    matriz TF-IDF del corpus completo, conjuntos de género/keywords/reparto por
    película, director normalizado, e índice invertido género -> posiciones.
    """
    print("Construyendo matriz TF-IDF y estructuras de similitud...")

    vectorizador = TfidfVectorizer(max_features=50_000, min_df=2, stop_words="english")
    tfidf_matrix = vectorizador.fit_transform(df["texto"])

    generos_sets = [_a_conjunto(c) for c in df["categoria"]]
    keywords_sets = [_a_conjunto(k) for k in df["keywords_limpio"]]
    cast_sets = [_a_conjunto(r) for r in df["reparto"]]
    directores = [str(d).strip().lower() for d in df["director"]]

    indice_invertido_genero: Dict[str, List[int]] = {}
    for posicion, generos in enumerate(generos_sets):
        for genero in generos:
            indice_invertido_genero.setdefault(genero, []).append(posicion)

    return EstructurasSimilitud(
        tfidf_matrix=tfidf_matrix,
        generos_sets=generos_sets,
        keywords_sets=keywords_sets,
        cast_sets=cast_sets,
        directores=directores,
        indice_invertido_genero=indice_invertido_genero,
    )


# ===============================
# 3. PARES POSITIVOS
# ===============================
def generar_pares_positivos(
    df: pd.DataFrame,
    estructuras: EstructurasSimilitud,
    n_por_ancla: int = N_POSITIVOS_POR_ANCLA,
    semilla: int = SEMILLA_ALEATORIA,
) -> pd.DataFrame:
    """
    Para cada película ancla, busca candidatos que compartan al menos un género
    (usando el índice invertido, no comparación contra todo el corpus), calcula
    un puntaje combinado (género + keywords + director + reparto + TF-IDF) y
    conserva los `n_por_ancla` candidatos con mayor puntaje que superen los
    umbrales mínimos.
    """
    rng = np.random.default_rng(semilla)
    ids = df["id"].to_numpy()
    titulos = df["titulo"].to_numpy()
    n = len(df)
    filas = []

    for i in range(n):
        generos_i = estructuras.generos_sets[i]
        if not generos_i:
            continue  # sin género no aplica el criterio principal de bloqueo

        candidatos = set()
        for genero in generos_i:
            candidatos.update(estructuras.indice_invertido_genero.get(genero, []))
        candidatos.discard(i)
        if not candidatos:
            continue

        candidatos_lista = list(candidatos)
        if len(candidatos_lista) > MAX_CANDIDATOS_POSITIVOS_POR_ANCLA:
            candidatos_lista = list(
                rng.choice(candidatos_lista, size=MAX_CANDIDATOS_POSITIVOS_POR_ANCLA, replace=False)
            )

        similitudes = cosine_similarity(
            estructuras.tfidf_matrix[i], estructuras.tfidf_matrix[candidatos_lista]
        ).ravel()

        puntuados = []
        for posicion, j in enumerate(candidatos_lista):
            sim_tfidf = float(similitudes[posicion])
            if sim_tfidf < UMBRAL_TFIDF_MIN_POSITIVO:
                continue

            jaccard_genero = _jaccard(generos_i, estructuras.generos_sets[j])
            jaccard_keywords = _jaccard(estructuras.keywords_sets[i], estructuras.keywords_sets[j])
            jaccard_reparto = _jaccard(estructuras.cast_sets[i], estructuras.cast_sets[j])
            mismo_director = bool(estructuras.directores[i]) and (
                estructuras.directores[i] == estructuras.directores[j]
            )

            score = (
                PESO_GENERO * jaccard_genero
                + PESO_KEYWORDS * jaccard_keywords
                + PESO_DIRECTOR * float(mismo_director)
                + PESO_REPARTO * jaccard_reparto
                + PESO_TFIDF * sim_tfidf
            )
            if score < UMBRAL_SCORE_POSITIVO:
                continue

            puntuados.append((j, score, jaccard_genero, jaccard_keywords, mismo_director, jaccard_reparto, sim_tfidf))

        puntuados.sort(key=lambda t: t[1], reverse=True)
        for j, score, jg, jk, md, jr, st in puntuados[:n_por_ancla]:
            filas.append({
                "id_ancla": ids[i],
                "titulo_ancla": titulos[i],
                "id_positivo": ids[j],
                "titulo_positivo": titulos[j],
                "score": round(score, 4),
                "jaccard_genero": round(jg, 4),
                "jaccard_keywords": round(jk, 4),
                "mismo_director": md,
                "jaccard_reparto": round(jr, 4),
                "similitud_tfidf": round(st, 4),
            })

    resultado = pd.DataFrame(filas)
    print(f"Pares positivos generados: {len(resultado)}")
    return resultado


# ===============================
# 4. PARES NEGATIVOS
# ===============================
def generar_pares_negativos(
    df: pd.DataFrame,
    estructuras: EstructurasSimilitud,
    n_por_ancla: int = N_NEGATIVOS_POR_ANCLA,
    semilla: int = SEMILLA_ALEATORIA,
) -> pd.DataFrame:
    """
    Para cada ancla, muestrea al azar un pool de candidatos (evita comparar
    contra todo el corpus) y se queda solo con los que cumplen TODOS los
    requisitos de disimilitud: género disjunto, casi ninguna keyword en común,
    baja similitud TF-IDF y distinto director. Así se evitan negativos ambiguos.
    """
    rng = np.random.default_rng(semilla)
    ids = df["id"].to_numpy()
    titulos = df["titulo"].to_numpy()
    n = len(df)
    todos_los_indices = np.arange(n)
    filas = []

    for i in range(n):
        generos_i = estructuras.generos_sets[i]

        tamano_muestra = min(MAX_MUESTREO_NEGATIVOS_POR_ANCLA, n - 1)
        muestra = rng.choice(todos_los_indices, size=tamano_muestra, replace=False)
        muestra = muestra[muestra != i]
        if len(muestra) == 0:
            continue

        similitudes = cosine_similarity(
            estructuras.tfidf_matrix[i], estructuras.tfidf_matrix[muestra]
        ).ravel()

        candidatos_validos = []
        for posicion, j in enumerate(muestra):
            jaccard_genero = _jaccard(generos_i, estructuras.generos_sets[j])
            if jaccard_genero > 0:
                continue  # géneros deben ser completamente disjuntos

            jaccard_keywords = _jaccard(estructuras.keywords_sets[i], estructuras.keywords_sets[j])
            if jaccard_keywords > UMBRAL_JACCARD_KEYWORDS_NEGATIVO:
                continue

            sim_tfidf = float(similitudes[posicion])
            if sim_tfidf > UMBRAL_TFIDF_MAX_NEGATIVO:
                continue

            mismo_director = bool(estructuras.directores[i]) and (
                estructuras.directores[i] == estructuras.directores[j]
            )
            if mismo_director:
                continue

            candidatos_validos.append((j, jaccard_genero, jaccard_keywords, sim_tfidf))

        if not candidatos_validos:
            continue

        rng.shuffle(candidatos_validos)
        for j, jg, jk, st in candidatos_validos[:n_por_ancla]:
            filas.append({
                "id_ancla": ids[i],
                "titulo_ancla": titulos[i],
                "id_negativo": ids[j],
                "titulo_negativo": titulos[j],
                "jaccard_genero": round(jg, 4),
                "jaccard_keywords": round(jk, 4),
                "similitud_tfidf": round(st, 4),
            })

    resultado = pd.DataFrame(filas)
    print(f"Pares negativos generados: {len(resultado)}")
    return resultado


# ===============================
# 5. TRIPLETAS (anchor, positive, negative)
# ===============================
def generar_tripletas(
    df: pd.DataFrame, positivos: pd.DataFrame, negativos: pd.DataFrame
) -> pd.DataFrame:
    """
    Combina cada positivo con un negativo de la misma ancla (ciclando si las
    cantidades no coinciden) para formar tripletas listas para triplet loss.
    Solo se generan tripletas para anclas que tengan al menos un positivo y
    un negativo válidos.
    """
    if positivos.empty or negativos.empty:
        print("No hay suficientes pares para construir tripletas.")
        return pd.DataFrame()

    texto_por_id = dict(zip(df["id"], df["texto"]))
    negativos_por_ancla = negativos.groupby("id_ancla")
    filas = []

    for id_ancla, grupo_pos in positivos.groupby("id_ancla"):
        if id_ancla not in negativos_por_ancla.groups:
            continue

        grupo_neg = negativos_por_ancla.get_group(id_ancla).reset_index(drop=True)
        grupo_pos = grupo_pos.reset_index(drop=True)

        for k in range(len(grupo_pos)):
            fila_pos = grupo_pos.iloc[k]
            fila_neg = grupo_neg.iloc[k % len(grupo_neg)]  # cicla si faltan negativos

            filas.append({
                "anchor_id": id_ancla,
                "anchor_titulo": fila_pos["titulo_ancla"],
                "anchor_texto": texto_por_id.get(id_ancla, ""),
                "positive_id": fila_pos["id_positivo"],
                "positive_titulo": fila_pos["titulo_positivo"],
                "positive_texto": texto_por_id.get(fila_pos["id_positivo"], ""),
                "negative_id": fila_neg["id_negativo"],
                "negative_titulo": fila_neg["titulo_negativo"],
                "negative_texto": texto_por_id.get(fila_neg["id_negativo"], ""),
                "score_positivo": fila_pos["score"],
                "similitud_tfidf_negativo": fila_neg["similitud_tfidf"],
            })

    resultado = pd.DataFrame(filas)
    print(f"Tripletas generadas: {len(resultado)}")
    return resultado


# ===============================
# 6. GUARDAR SALIDAS
# ===============================
def guardar_dataset_entrenamiento(
    positivos: pd.DataFrame, negativos: pd.DataFrame, tripletas: pd.DataFrame
) -> None:
    """
    Guarda los 4 artefactos de esta etapa en data/training/:
      - pares_positivos.csv / pares_negativos.csv: pares con su justificación (scores).
      - tripletas.csv: vista legible (ids, títulos y métricas) de cada tripleta.
      - dataset_entrenamiento.csv: solo las 3 columnas de texto (anchor, positive,
        negative), listas para alimentar el fine-tuning de SentenceTransformers
        en la siguiente etapa.
    """
    DATA_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    positivos.to_csv(DATA_TRAINING_DIR / "pares_positivos.csv", index=False)
    negativos.to_csv(DATA_TRAINING_DIR / "pares_negativos.csv", index=False)

    if not tripletas.empty:
        columnas_legibles = [
            "anchor_id", "anchor_titulo",
            "positive_id", "positive_titulo",
            "negative_id", "negative_titulo",
            "score_positivo", "similitud_tfidf_negativo",
        ]
        tripletas[columnas_legibles].to_csv(DATA_TRAINING_DIR / "tripletas.csv", index=False)

        dataset_entrenamiento = tripletas[["anchor_texto", "positive_texto", "negative_texto"]].rename(
            columns={
                "anchor_texto": "anchor",
                "positive_texto": "positive",
                "negative_texto": "negative",
            }
        )
        dataset_entrenamiento.to_csv(DATA_TRAINING_DIR / "dataset_entrenamiento.csv", index=False)
    else:
        pd.DataFrame(columns=["anchor_id", "anchor_titulo", "positive_id", "titulo_positivo"]).to_csv(
            DATA_TRAINING_DIR / "tripletas.csv", index=False
        )
        pd.DataFrame(columns=["anchor", "positive", "negative"]).to_csv(
            DATA_TRAINING_DIR / "dataset_entrenamiento.csv", index=False
        )

    print(f"Archivos guardados en: {DATA_TRAINING_DIR}")


# ===============================
# ORQUESTACIÓN
# ===============================
def main() -> None:
    df = cargar_dataset()
    df = crear_columna_texto(df)

    estructuras = _construir_estructuras_similitud(df)

    positivos = generar_pares_positivos(df, estructuras)
    negativos = generar_pares_negativos(df, estructuras)
    tripletas = generar_tripletas(df, positivos, negativos)

    guardar_dataset_entrenamiento(positivos, negativos, tripletas)

    print("\n" + "=" * 60)
    print("ETAPA 2 COMPLETADA: dataset de entrenamiento (pares y tripletas) listo.")
    print("(Todavía sin embeddings, sin ChromaDB, sin fine-tuning, sin cambios en FastAPI)")
    print("=" * 60)


if __name__ == "__main__":
    main()
