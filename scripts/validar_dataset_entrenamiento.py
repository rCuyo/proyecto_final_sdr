"""
Validación, estadísticas, visualizaciones, mejora de cobertura y Hard
Negative Mining del dataset de entrenamiento para el Fine-Tuning de BAAI/bge-m3.

Esta es la ETAPA 3 del proyecto. Reutiliza el dataset limpio de la etapa 1
(`peliculas_dataset_limpio.csv`) y los pares/tripletas de la etapa 2
(`data/training/*.csv`), y reutiliza directamente varias funciones y
constantes de `generar_dataset_entrenamiento.py` (carga, columna 'texto',
estructuras de similitud TF-IDF, emparejador de tripletas y los mismos pesos
del puntaje combinado) en lugar de reimplementarlas.

Todavía NO se generan embeddings, NO se usa SentenceTransformer/torch/
transformers, NO se construye ChromaDB y NO se toca FastAPI. La similitud
semántica se aproxima únicamente con TF-IDF + similitud coseno.

Requiere:
    pip install pandas numpy scikit-learn matplotlib
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # backend sin GUI: solo necesitamos guardar PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Reutilización explícita de la etapa 2 (no reimplementamos esta lógica).
# No se importa `preparar_dataset.py` porque este script no descarga nada
# de Kaggle y no queremos arrastrar esa dependencia.
from generar_dataset_entrenamiento import (
    MAX_CANDIDATOS_POSITIVOS_POR_ANCLA,
    PESO_DIRECTOR,
    PESO_GENERO,
    PESO_KEYWORDS,
    PESO_REPARTO,
    PESO_TFIDF,
    UMBRAL_TFIDF_MAX_NEGATIVO,
    UMBRAL_TFIDF_MIN_POSITIVO,
    EstructurasSimilitud,
    _construir_estructuras_similitud,
    _jaccard,
    cargar_dataset,
    crear_columna_texto,
    generar_tripletas as generar_tripletas_base,
)

# ===============================
# CONFIGURACIÓN
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_PELICULAS = BASE_DIR / "data" / "processed" / "peliculas_dataset_limpio.csv"
DIR_TRAINING = BASE_DIR / "data" / "training"

RUTA_POSITIVOS = DIR_TRAINING / "pares_positivos.csv"
RUTA_NEGATIVOS = DIR_TRAINING / "pares_negativos.csv"
RUTA_TRIPLETAS = DIR_TRAINING / "tripletas.csv"
RUTA_DATASET_ENTRENAMIENTO = DIR_TRAINING / "dataset_entrenamiento.csv"

DIR_REPORTS = BASE_DIR / "data" / "reports"
DIR_GRAFICOS = DIR_REPORTS / "graficos"
RUTA_REPORTE_TXT = DIR_REPORTS / "dataset_report.txt"
RUTA_ESTADISTICAS_JSON = DIR_REPORTS / "dataset_statistics.json"

# Artefactos propios de esta etapa (se regeneran por completo en cada corrida;
# los de la etapa 2 -arriba- nunca se sobrescriben).
RUTA_POSITIVOS_MEJORADO = DIR_TRAINING / "pares_positivos_mejorado.csv"
RUTA_TRIPLETAS_HARD = DIR_TRAINING / "tripletas_hard.csv"
RUTA_DATASET_V2 = DIR_TRAINING / "dataset_entrenamiento_v2.csv"

TOP_N_REPORTE = 20

# --- Cobertura de positivos (fallback Top-1) ---
# Umbral más bajo que UMBRAL_SCORE_POSITIVO (0.35) de la etapa 2, pero NO es
# "bajar el umbral global": solo se usa para películas que ya se sabe que no
# alcanzan el umbral estricto con NINGÚN candidato, y ahí se toma el MEJOR
# candidato de esa película puntual (Top-1 de su propio ranking), no el
# primero que supere un umbral relajado aplicado a todos por igual.
UMBRAL_SCORE_POSITIVO_FALLBACK = 0.20

# --- Hard Negative Mining ---
# El techo por tripleta es sim(ancla,positivo) - MARGEN_HARD_NEGATIVO (no un
# valor fijo global): garantiza sim(ancla,positivo) > sim(ancla,hard_negativo).
# El margen es chico porque las similitudes TF-IDF reales de este corpus son
# bajas (el positivo promedio ronda ~0.13): un margen grande (p.ej. 0.10)
# dejaría la banda vacía para la mayoría de los pares.
MARGEN_HARD_NEGATIVO = 0.02
# Piso: el mismo techo usado para negativos fáciles en la etapa 2
# (UMBRAL_TFIDF_MAX_NEGATIVO). Así "más difícil que un negativo fácil" es un
# criterio literal y no un número mágico adicional.
UMBRAL_HARD_NEG_SIM_MAX = 0.95  # techo absoluto: excluye casi-duplicados/remakes
CANDIDATOS_A_EXAMINAR_POR_ANCLA = 300  # top-K más parecidos antes de filtrar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validar_dataset_entrenamiento")


# ===============================
# Tipos de datos
# ===============================
@dataclass
class Incidencia:
    """Un hallazgo de la validación: nivel ('ERROR' | 'ADVERTENCIA' | 'INFO'), categoría y mensaje."""

    nivel: str
    categoria: str
    mensaje: str


# ===============================
# 0. CARGA DE DATOS
# ===============================
def cargar_todos_los_datos(
    ruta_peliculas: Path = RUTA_PELICULAS,
    ruta_positivos: Path = RUTA_POSITIVOS,
    ruta_negativos: Path = RUTA_NEGATIVOS,
    ruta_tripletas: Path = RUTA_TRIPLETAS,
    ruta_dataset_entrenamiento: Path = RUTA_DATASET_ENTRENAMIENTO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga las películas (reutilizando `cargar_dataset` + `crear_columna_texto`
    de la etapa 2) y los 4 artefactos de entrenamiento de la etapa 2.

    Lanza `FileNotFoundError` con un mensaje claro si falta algún archivo,
    indicando qué script anterior hay que ejecutar primero.
    """
    try:
        peliculas = cargar_dataset(ruta_peliculas)
        peliculas = crear_columna_texto(peliculas)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}. Ejecuta primero scripts/preparar_dataset.py."
        ) from error

    rutas_entrenamiento = {
        "pares_positivos.csv": ruta_positivos,
        "pares_negativos.csv": ruta_negativos,
        "tripletas.csv": ruta_tripletas,
        "dataset_entrenamiento.csv": ruta_dataset_entrenamiento,
    }
    faltantes = [nombre for nombre, ruta in rutas_entrenamiento.items() if not ruta.exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan archivos de entrenamiento {faltantes} en {DIR_TRAINING}. "
            "Ejecuta primero scripts/generar_dataset_entrenamiento.py."
        )

    positivos = pd.read_csv(ruta_positivos)
    negativos = pd.read_csv(ruta_negativos)
    tripletas = pd.read_csv(ruta_tripletas)
    dataset_entrenamiento = pd.read_csv(ruta_dataset_entrenamiento)

    logger.info(
        "Datos cargados: %d películas, %d positivos, %d negativos, %d tripletas.",
        len(peliculas), len(positivos), len(negativos), len(tripletas),
    )
    return peliculas, positivos, negativos, tripletas, dataset_entrenamiento


# ===============================
# COBERTURA DE POSITIVOS (fallback Top-1)
# ===============================
def generar_positivos_fallback(
    peliculas: pd.DataFrame,
    estructuras: EstructurasSimilitud,
    positivos_existentes: pd.DataFrame,
    umbral_minimo: float = UMBRAL_SCORE_POSITIVO_FALLBACK,
) -> pd.DataFrame:
    """
    Mejora la cobertura de positivos sin bajar el umbral global de la etapa 2.

    Criterio: para cada película que quedó sin ningún positivo (todos sus
    candidatos por debajo de UMBRAL_SCORE_POSITIVO, o directamente sin
    candidatos), se reutiliza el mismo puntaje combinado de la etapa 2
    (mismos pesos: género + keywords + director + reparto + TF-IDF) sobre el
    mismo pool de candidatos (bloqueo por género compartido), pero en vez de
    exigir el umbral estricto (0.35) se toma el candidato de MEJOR puntaje
    disponible para ESA película ("Top-1" de su propio ranking) y se acepta
    solo si supera un piso de calidad más bajo (UMBRAL_SCORE_POSITIVO_FALLBACK).

    Esto es distinto de simplemente bajar el umbral global: cada película
    compite contra su propio mejor candidato, no contra un umbral relajado
    aplicado a todo el dataset por igual. Una película sin señal temática real
    (sin género, sin candidatos que compartan género, o cuyo mejor candidato
    ni así alcanza el piso mínimo) sigue quedando sin positivo: no se fuerza
    un emparejamiento irrelevante solo para completar la cobertura.
    """
    ids = peliculas["id"].to_numpy()
    titulos = peliculas["titulo"].to_numpy()

    ids_ya_cubiertos = set(positivos_existentes["id_ancla"]) if not positivos_existentes.empty else set()
    posiciones_sin_positivo = [i for i, pid in enumerate(ids) if pid not in ids_ya_cubiertos]

    logger.info("Buscando positivo de respaldo para %d películas sin cobertura...", len(posiciones_sin_positivo))

    filas = []
    sin_candidato_valido = 0

    for contador, i in enumerate(posiciones_sin_positivo, start=1):
        if contador % 5000 == 0:
            logger.info("Positivos de respaldo: %d/%d procesadas...", contador, len(posiciones_sin_positivo))

        generos_i = estructuras.generos_sets[i]
        if not generos_i:
            sin_candidato_valido += 1
            continue

        candidatos = set()
        for genero in generos_i:
            candidatos.update(estructuras.indice_invertido_genero.get(genero, []))
        candidatos.discard(i)
        if not candidatos:
            sin_candidato_valido += 1
            continue

        candidatos_lista = list(candidatos)
        if len(candidatos_lista) > MAX_CANDIDATOS_POSITIVOS_POR_ANCLA:
            candidatos_lista = candidatos_lista[:MAX_CANDIDATOS_POSITIVOS_POR_ANCLA]

        similitudes = cosine_similarity(
            estructuras.tfidf_matrix[i], estructuras.tfidf_matrix[candidatos_lista]
        ).ravel()

        mejor = None  # (score, j, jaccard_genero, jaccard_keywords, mismo_director, jaccard_reparto, sim_tfidf)
        for posicion, j in enumerate(candidatos_lista):
            sim_tfidf = float(similitudes[posicion])
            if sim_tfidf < UMBRAL_TFIDF_MIN_POSITIVO:
                continue  # se mantiene la exigencia de relación textual mínima real

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

            if mejor is None or score > mejor[0]:
                mejor = (score, j, jaccard_genero, jaccard_keywords, mismo_director, jaccard_reparto, sim_tfidf)

        if mejor is None or mejor[0] < umbral_minimo:
            sin_candidato_valido += 1
            continue

        score, j, jg, jk, md, jr, st = mejor
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

    if sin_candidato_valido:
        logger.warning(
            "%d películas siguen sin positivo válido (sin género, sin candidatos, o mejor score bajo %.2f).",
            sin_candidato_valido, umbral_minimo,
        )

    resultado = pd.DataFrame(filas)
    logger.info("Positivos de respaldo generados: %d", len(resultado))
    return resultado


def combinar_positivos(positivos_originales: pd.DataFrame, positivos_fallback: pd.DataFrame) -> pd.DataFrame:
    """Une los positivos originales de la etapa 2 con los de respaldo, marcando el origen de cada fila."""
    originales = positivos_originales.copy()
    originales["origen"] = "original"

    if positivos_fallback.empty:
        return originales

    fallback = positivos_fallback.copy()
    fallback["origen"] = "fallback_top1"
    return pd.concat([originales, fallback], ignore_index=True)


def guardar_positivos_mejorado(positivos_mejorado: pd.DataFrame, ruta: Path = RUTA_POSITIVOS_MEJORADO) -> None:
    """Guarda los positivos combinados (originales + respaldo) sin tocar pares_positivos.csv de la etapa 2."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    positivos_mejorado.to_csv(ruta, index=False)
    logger.info("Positivos mejorados guardados en: %s (%d filas)", ruta, len(positivos_mejorado))


# ===============================
# 1. VALIDACIÓN
# ===============================
def validar_conteos(
    peliculas: pd.DataFrame, positivos: pd.DataFrame, negativos: pd.DataFrame, tripletas: pd.DataFrame
) -> List[Incidencia]:
    """Registra los conteos base y marca error si algún artefacto quedó vacío."""
    incidencias = [
        Incidencia("INFO", "conteos", f"Películas: {len(peliculas)}"),
        Incidencia("INFO", "conteos", f"Pares positivos: {len(positivos)}"),
        Incidencia("INFO", "conteos", f"Pares negativos: {len(negativos)}"),
        Incidencia("INFO", "conteos", f"Tripletas: {len(tripletas)}"),
    ]
    if len(positivos) == 0:
        incidencias.append(Incidencia("ERROR", "conteos", "No existen pares positivos."))
    if len(negativos) == 0:
        incidencias.append(Incidencia("ERROR", "conteos", "No existen pares negativos."))
    if len(tripletas) == 0:
        incidencias.append(Incidencia("ERROR", "conteos", "No existen tripletas."))
    return incidencias


def validar_campos_vacios(peliculas: pd.DataFrame) -> List[Incidencia]:
    """
    Detecta textos/títulos/categorías/keywords/directores/actores vacíos.
    Título y texto vacíos son ERROR (la fila es inutilizable); el resto son
    ADVERTENCIA (reduce la calidad de la señal, pero la fila sigue sirviendo).
    """
    incidencias = []
    columnas_y_nivel = {
        "titulo": "ERROR",
        "texto": "ERROR",
        "categoria": "ADVERTENCIA",
        "keywords_limpio": "ADVERTENCIA",
        "director": "ADVERTENCIA",
        "reparto": "ADVERTENCIA",
    }
    for columna, nivel in columnas_y_nivel.items():
        vacios = peliculas[columna].fillna("").astype(str).str.strip().eq("").sum()
        if vacios > 0:
            incidencias.append(
                Incidencia(nivel, "campos_vacios", f"{vacios} películas con '{columna}' vacío.")
            )
    return incidencias


def validar_duplicados_peliculas(peliculas: pd.DataFrame) -> List[Incidencia]:
    """Detecta filas totalmente duplicadas e IDs de película duplicados."""
    incidencias = []
    duplicados_totales = int(peliculas.duplicated().sum())
    if duplicados_totales:
        incidencias.append(
            Incidencia("ADVERTENCIA", "duplicados", f"{duplicados_totales} filas de película totalmente duplicadas.")
        )
    ids_duplicados = int(peliculas["id"].duplicated().sum())
    if ids_duplicados:
        incidencias.append(Incidencia("ERROR", "duplicados", f"{ids_duplicados} IDs de película duplicados."))
    return incidencias


def validar_tripletas_duplicadas(tripletas: pd.DataFrame) -> List[Incidencia]:
    """Detecta tripletas repetidas (mismo anchor, positive y negative)."""
    if tripletas.empty:
        return []
    duplicadas = int(tripletas.duplicated(subset=["anchor_id", "positive_id", "negative_id"]).sum())
    if duplicadas:
        return [Incidencia("ADVERTENCIA", "tripletas_duplicadas", f"{duplicadas} tripletas duplicadas.")]
    return []


def validar_integridad_tripletas(tripletas: pd.DataFrame) -> List[Incidencia]:
    """Detecta tripletas inválidas: anchor==positive, positive==negative o anchor==negative."""
    if tripletas.empty:
        return []
    incidencias = []
    casos = {
        "anchor == positive": (tripletas["anchor_id"] == tripletas["positive_id"]).sum(),
        "positive == negative": (tripletas["positive_id"] == tripletas["negative_id"]).sum(),
        "anchor == negative": (tripletas["anchor_id"] == tripletas["negative_id"]).sum(),
    }
    for descripcion, cantidad in casos.items():
        if cantidad:
            incidencias.append(
                Incidencia("ERROR", "integridad_tripletas", f"{int(cantidad)} tripletas con {descripcion}.")
            )
    return incidencias


def calcular_cobertura(
    peliculas: pd.DataFrame, positivos: pd.DataFrame, negativos: pd.DataFrame, tripletas: pd.DataFrame
) -> Dict[str, int]:
    """Cuenta películas sin ningún positivo/negativo/tripleta (estructurado, para JSON y comparación entre corridas)."""
    ids_todas = set(peliculas["id"])
    ids_con_positivo = set(positivos["id_ancla"]) if not positivos.empty else set()
    ids_con_negativo = set(negativos["id_ancla"]) if not negativos.empty else set()
    ids_con_tripleta = set(tripletas["anchor_id"]) if not tripletas.empty else set()

    return {
        "peliculas_sin_positivo": len(ids_todas - ids_con_positivo),
        "peliculas_sin_negativo": len(ids_todas - ids_con_negativo),
        "peliculas_sin_tripleta": len(ids_todas - ids_con_tripleta),
    }


def validar_cobertura(
    peliculas: pd.DataFrame, positivos: pd.DataFrame, negativos: pd.DataFrame, tripletas: pd.DataFrame
) -> List[Incidencia]:
    """Cuenta películas que quedaron sin ningún positivo, negativo o tripleta."""
    cobertura = calcular_cobertura(peliculas, positivos, negativos, tripletas)
    return [
        Incidencia("ADVERTENCIA", "cobertura", f"{cobertura['peliculas_sin_positivo']} películas sin ningún par positivo."),
        Incidencia("ADVERTENCIA", "cobertura", f"{cobertura['peliculas_sin_negativo']} películas sin ningún par negativo."),
        Incidencia("ADVERTENCIA", "cobertura", f"{cobertura['peliculas_sin_tripleta']} películas sin ninguna tripleta."),
    ]


def ejecutar_validaciones(
    peliculas: pd.DataFrame, positivos: pd.DataFrame, negativos: pd.DataFrame, tripletas: pd.DataFrame
) -> List[Incidencia]:
    """Ejecuta todos los chequeos de la Parte 1 y devuelve la lista combinada de incidencias."""
    incidencias: List[Incidencia] = []
    incidencias += validar_conteos(peliculas, positivos, negativos, tripletas)
    incidencias += validar_campos_vacios(peliculas)
    incidencias += validar_duplicados_peliculas(peliculas)
    incidencias += validar_tripletas_duplicadas(tripletas)
    incidencias += validar_integridad_tripletas(tripletas)
    incidencias += validar_cobertura(peliculas, positivos, negativos, tripletas)

    errores = sum(1 for i in incidencias if i.nivel == "ERROR")
    advertencias = sum(1 for i in incidencias if i.nivel == "ADVERTENCIA")
    logger.info("Validación completa: %d errores, %d advertencias.", errores, advertencias)
    return incidencias


# ===============================
# 2. ESTADÍSTICAS
# ===============================
def _dividir_lista(cadena: Any) -> List[str]:
    """Convierte 'Action, Drama' -> ['Action', 'Drama']. Preserva mayúsculas (para reportes legibles)."""
    if not isinstance(cadena, str) or not cadena.strip():
        return []
    return [t.strip() for t in cadena.split(",") if t.strip()]


def _contador_a_lista(contador: Counter, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convierte un Counter en una lista de {'nombre':..., 'cantidad':...} apta para JSON/reportes."""
    items = contador.most_common(top_n)
    return [{"nombre": nombre, "cantidad": cantidad} for nombre, cantidad in items]


def calcular_estadisticas(peliculas: pd.DataFrame, tripletas: pd.DataFrame) -> Dict[str, Any]:
    """Calcula todas las estadísticas pedidas en la Parte 2 a partir del dataset de películas."""
    generos_todos: List[str] = []
    keywords_todas: List[str] = []
    actores_todos: List[str] = []

    for cadena in peliculas["categoria"]:
        generos_todos.extend(_dividir_lista(cadena))
    for cadena in peliculas["keywords_limpio"]:
        keywords_todas.extend(_dividir_lista(cadena))
    for cadena in peliculas["reparto"]:
        actores_todos.extend(_dividir_lista(cadena))

    directores_todos = [d.strip() for d in peliculas["director"] if isinstance(d, str) and d.strip()]

    contador_generos = Counter(generos_todos)
    contador_keywords = Counter(keywords_todas)
    contador_directores = Counter(directores_todos)
    contador_actores = Counter(actores_todos)

    numero_keywords_por_pelicula = peliculas["keywords_limpio"].apply(lambda c: len(_dividir_lista(c)))
    numero_actores_por_pelicula = peliculas["reparto"].apply(lambda c: len(_dividir_lista(c)))
    longitudes_texto = peliculas["texto"].str.len()

    return {
        "numero_peliculas": int(len(peliculas)),
        "numero_generos": int(len(contador_generos)),
        "distribucion_generos": dict(contador_generos.most_common()),
        "promedio_keywords_por_pelicula": float(numero_keywords_por_pelicula.mean()),
        "promedio_actores_por_pelicula": float(numero_actores_por_pelicula.mean()),
        "longitud_promedio_texto": float(longitudes_texto.mean()),
        "top_generos": _contador_a_lista(contador_generos, TOP_N_REPORTE),
        "top_keywords": _contador_a_lista(contador_keywords, TOP_N_REPORTE),
        "top_directores": _contador_a_lista(contador_directores, TOP_N_REPORTE),
        "top_actores": _contador_a_lista(contador_actores, TOP_N_REPORTE),
        "cantidad_tripletas": int(len(tripletas)),
    }


def agregar_resumen_validacion(estadisticas: Dict[str, Any], incidencias: List[Incidencia]) -> Dict[str, Any]:
    """Devuelve una copia de `estadisticas` con el conteo de errores/advertencias de la validación."""
    resultado = dict(estadisticas)
    resultado["cantidad_errores"] = sum(1 for i in incidencias if i.nivel == "ERROR")
    resultado["cantidad_advertencias"] = sum(1 for i in incidencias if i.nivel == "ADVERTENCIA")
    return resultado


def cargar_estadisticas_anteriores(ruta: Path = RUTA_ESTADISTICAS_JSON) -> Optional[Dict[str, Any]]:
    """Carga el dataset_statistics.json de la corrida anterior (si existe) ANTES de sobrescribirlo, para comparar."""
    if not ruta.exists():
        return None
    with ruta.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_estadisticas_json(estadisticas: Dict[str, Any], ruta: Path = RUTA_ESTADISTICAS_JSON) -> None:
    """Guarda las estadísticas (con el resumen de validación ya incluido) como JSON."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(estadisticas, archivo, ensure_ascii=False, indent=2)
    logger.info("Estadísticas guardadas en: %s", ruta)


def _formatear_top(lista: List[Dict[str, Any]]) -> str:
    return "\n".join(f"  {i+1:>2}. {item['nombre']} ({item['cantidad']})" for i, item in enumerate(lista))


def _formatear_valor(valor: Any) -> str:
    """Formatea un valor para la tabla de comparación; 'N/D' si falta en la corrida anterior."""
    if valor is None:
        return "N/D"
    if isinstance(valor, float):
        return f"{valor:.4f}"
    return str(valor)


def construir_comparacion(anteriores: Optional[Dict[str, Any]], actuales: Dict[str, Any]) -> str:
    """
    Compara cobertura, cantidades y similitudes entre la ejecución anterior
    (leída de dataset_statistics.json antes de sobrescribirlo) y la actual.
    """
    if anteriores is None:
        return "No se encontró una ejecución anterior (dataset_statistics.json) con la cual comparar."

    eval_ant = anteriores.get("evaluacion_hard_negative_mining", {})
    eval_act = actuales.get("evaluacion_hard_negative_mining", {})
    cob_ant = anteriores.get("cobertura", {})
    cob_act = actuales.get("cobertura", {})

    filas = [
        ("Películas sin positivo", cob_ant.get("peliculas_sin_positivo"), cob_act.get("peliculas_sin_positivo")),
        ("Películas sin tripleta", cob_ant.get("peliculas_sin_tripleta"), cob_act.get("peliculas_sin_tripleta")),
        ("Cantidad de tripletas (fáciles)", eval_ant.get("cantidad_tripletas_originales"), eval_act.get("cantidad_tripletas_originales")),
        ("Cantidad de hard negatives / tripletas hard", eval_ant.get("cantidad_tripletas_hard"), eval_act.get("cantidad_tripletas_hard")),
        ("Similitud promedio ancla-positivo", eval_ant.get("similitud_promedio_anchor_positivo"), eval_act.get("similitud_promedio_anchor_positivo")),
        ("Similitud promedio ancla-negativo (fácil)", eval_ant.get("similitud_promedio_anchor_negativo_facil"), eval_act.get("similitud_promedio_anchor_negativo_facil")),
        ("Similitud promedio ancla-negativo (hard)", eval_ant.get("similitud_promedio_anchor_negativo_hard"), eval_act.get("similitud_promedio_anchor_negativo_hard")),
        ("Brecha positivo-negativo (hard)", eval_ant.get("brecha_positivo_negativo_hard"), eval_act.get("brecha_positivo_negativo_hard")),
    ]

    lineas = [f"{'Métrica':<45}{'Antes':>15}{'Ahora':>15}"]
    for nombre, antes, ahora in filas:
        lineas.append(f"{nombre:<45}{_formatear_valor(antes):>15}{_formatear_valor(ahora):>15}")

    return "\n".join(lineas)


def construir_reporte_texto(
    estadisticas: Dict[str, Any],
    incidencias: List[Incidencia],
    evaluacion: Dict[str, Any],
    comparacion: str,
) -> str:
    """Construye el reporte de texto plano con validación, estadísticas, evaluación y comparación entre corridas."""
    lineas = []
    lineas.append("=" * 60)
    lineas.append("REPORTE DE VALIDACIÓN Y ESTADÍSTICAS DEL DATASET")
    lineas.append("=" * 60)

    lineas.append("\n--- 1. VALIDACIÓN ---")
    lineas.append(f"Errores encontrados: {estadisticas.get('cantidad_errores', 0)}")
    lineas.append(f"Advertencias encontradas: {estadisticas.get('cantidad_advertencias', 0)}")
    for incidencia in incidencias:
        if incidencia.nivel in ("ERROR", "ADVERTENCIA"):
            lineas.append(f"  [{incidencia.nivel}] ({incidencia.categoria}) {incidencia.mensaje}")

    lineas.append("\n--- 2. ESTADÍSTICAS GENERALES ---")
    lineas.append(f"Número de películas: {estadisticas['numero_peliculas']}")
    lineas.append(f"Número de géneros distintos: {estadisticas['numero_generos']}")
    lineas.append(f"Promedio de keywords por película: {estadisticas['promedio_keywords_por_pelicula']:.2f}")
    lineas.append(f"Promedio de actores por película: {estadisticas['promedio_actores_por_pelicula']:.2f}")
    lineas.append(f"Longitud promedio del texto (caracteres): {estadisticas['longitud_promedio_texto']:.1f}")
    lineas.append(f"Cantidad de tripletas: {estadisticas['cantidad_tripletas']}")

    lineas.append(f"\nTop {TOP_N_REPORTE} géneros:")
    lineas.append(_formatear_top(estadisticas["top_generos"]))
    lineas.append(f"\nTop {TOP_N_REPORTE} keywords:")
    lineas.append(_formatear_top(estadisticas["top_keywords"]))
    lineas.append(f"\nTop {TOP_N_REPORTE} directores:")
    lineas.append(_formatear_top(estadisticas["top_directores"]))
    lineas.append(f"\nTop {TOP_N_REPORTE} actores:")
    lineas.append(_formatear_top(estadisticas["top_actores"]))

    lineas.append("\n--- 3. HARD NEGATIVE MINING: EVALUACIÓN ---")
    lineas.append(f"Tripletas fáciles: {evaluacion['cantidad_tripletas_originales']}")
    lineas.append(f"Tripletas hard: {evaluacion['cantidad_tripletas_hard']}")
    lineas.append(f"Similitud promedio ancla-positivo (todos los positivos): {evaluacion['similitud_promedio_anchor_positivo']:.4f}")
    lineas.append(
        f"Similitud promedio ancla-positivo (solo pares con hard negative): "
        f"{evaluacion['similitud_promedio_anchor_positivo_subset_hard']:.4f}"
    )
    lineas.append(f"Similitud promedio ancla-negativo (fácil): {evaluacion['similitud_promedio_anchor_negativo_facil']:.4f}")
    lineas.append(f"Similitud promedio ancla-negativo (hard): {evaluacion['similitud_promedio_anchor_negativo_hard']:.4f}")
    lineas.append(f"Brecha positivo-negativo (fácil): {evaluacion['brecha_positivo_negativo_facil']:.4f}")
    lineas.append(f"Brecha positivo-negativo (hard): {evaluacion['brecha_positivo_negativo_hard']:.4f}")
    lineas.append(
        "\nInterpretación: por construcción, cada tripleta hard cumple "
        f"similitud(ancla,positivo) > similitud(ancla,hard_negativo) + {MARGEN_HARD_NEGATIVO} "
        "(margen configurable), así que la brecha (hard) siempre debería ser positiva. "
        "Cuanto más chica esa brecha -sin llegar a negativa-, más le cuesta al modelo "
        "separar ancla de negativo, y más útil es la tripleta para el entrenamiento."
    )

    lineas.append("\n--- 4. COMPARACIÓN CON LA EJECUCIÓN ANTERIOR ---")
    lineas.append(comparacion)

    return "\n".join(lineas)


def guardar_reporte_texto(texto: str, ruta: Path = RUTA_REPORTE_TXT) -> None:
    """Guarda el reporte de texto plano en disco."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    logger.info("Reporte de texto guardado en: %s", ruta)


# ===============================
# 3. VISUALIZACIONES
# ===============================
def _graficar_barras(items: List[Dict[str, Any]], titulo: str, ruta: Path) -> None:
    """Bar chart genérico para listas de {'nombre','cantidad'} (géneros/keywords/directores)."""
    if not items:
        logger.warning("Sin datos para graficar: %s", titulo)
        return
    nombres = [item["nombre"] for item in items]
    cantidades = [item["cantidad"] for item in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(nombres, cantidades, color="#4C72B0")
    ax.set_title(titulo)
    ax.set_ylabel("Cantidad de películas")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("Gráfico guardado en: %s", ruta)


def graficar_top_generos(estadisticas: Dict[str, Any], ruta: Path = DIR_GRAFICOS / "hist_generos.png") -> None:
    """Bar chart con el Top-N de géneros por cantidad de películas."""
    _graficar_barras(estadisticas["top_generos"], f"Top {TOP_N_REPORTE} géneros", ruta)


def graficar_top_keywords(estadisticas: Dict[str, Any], ruta: Path = DIR_GRAFICOS / "hist_keywords.png") -> None:
    """Bar chart con el Top-N de keywords por frecuencia."""
    _graficar_barras(estadisticas["top_keywords"], f"Top {TOP_N_REPORTE} keywords", ruta)


def graficar_top_directores(estadisticas: Dict[str, Any], ruta: Path = DIR_GRAFICOS / "hist_directores.png") -> None:
    """Bar chart con el Top-N de directores por cantidad de películas."""
    _graficar_barras(estadisticas["top_directores"], f"Top {TOP_N_REPORTE} directores", ruta)


def graficar_longitud_textos(peliculas: pd.DataFrame, ruta: Path = DIR_GRAFICOS / "hist_longitud_textos.png") -> None:
    """Histograma real (numérico) de la longitud en caracteres de la columna 'texto'."""
    longitudes = peliculas["texto"].str.len()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(longitudes, bins=40, color="#55A868")
    ax.set_title("Distribución de la longitud del texto por película")
    ax.set_xlabel("Longitud del texto (caracteres)")
    ax.set_ylabel("Cantidad de películas")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("Gráfico guardado en: %s", ruta)


def graficar_tripletas_por_ancla(tripletas: pd.DataFrame, ruta: Path = DIR_GRAFICOS / "hist_tripletas.png") -> None:
    """Histograma de cuántas tripletas aporta cada película-ancla (mide balance de cobertura)."""
    if tripletas.empty:
        logger.warning("Sin tripletas para graficar hist_tripletas.png")
        return
    conteos = tripletas.groupby("anchor_id").size()

    fig, ax = plt.subplots(figsize=(10, 6))
    max_valor = int(conteos.max())
    ax.hist(conteos, bins=range(1, max_valor + 2), align="left", color="#C44E52")
    ax.set_title("Cantidad de tripletas por película-ancla")
    ax.set_xlabel("Tripletas por ancla")
    ax.set_ylabel("Cantidad de anclas")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("Gráfico guardado en: %s", ruta)


def generar_graficos(
    peliculas: pd.DataFrame,
    tripletas: pd.DataFrame,
    estadisticas: Dict[str, Any],
    dir_graficos: Path = DIR_GRAFICOS,
) -> None:
    """Genera los 5 gráficos pedidos en la Parte 3 dentro de data/reports/graficos/."""
    dir_graficos.mkdir(parents=True, exist_ok=True)
    # Se pasa `ruta` explícitamente (en vez de confiar en los defaults de cada
    # función, que quedan fijados al valor de DIR_GRAFICOS al definirse el módulo).
    graficar_top_generos(estadisticas, dir_graficos / "hist_generos.png")
    graficar_top_keywords(estadisticas, dir_graficos / "hist_keywords.png")
    graficar_top_directores(estadisticas, dir_graficos / "hist_directores.png")
    graficar_longitud_textos(peliculas, dir_graficos / "hist_longitud_textos.png")
    graficar_tripletas_por_ancla(tripletas, dir_graficos / "hist_tripletas.png")


# ===============================
# 4. HARD NEGATIVE MINING
# ===============================
def _candidatos_ordenados_por_similitud(
    posicion_ancla: int, estructuras: EstructurasSimilitud
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Similitud TF-IDF del ancla contra todo el corpus (un único producto
    disperso) y los índices de los `CANDIDATOS_A_EXAMINAR_POR_ANCLA` más
    parecidos, ya ordenados descendentemente (vía argpartition, evitando
    ordenar los ~45k con argsort). Se calcula una sola vez por ancla y se
    reutiliza para cada uno de sus positivos (evita recomputar el producto
    disperso si el ancla tiene más de un positivo).
    """
    similitudes = cosine_similarity(
        estructuras.tfidf_matrix[posicion_ancla], estructuras.tfidf_matrix
    ).ravel()

    k = min(CANDIDATOS_A_EXAMINAR_POR_ANCLA, len(similitudes) - 1)
    indices_parciales = np.argpartition(-similitudes, k)[:k]
    indices_ordenados = indices_parciales[np.argsort(-similitudes[indices_parciales])]
    return indices_ordenados, similitudes


def _elegir_hard_negativo(
    posicion_ancla: int,
    estructuras: EstructurasSimilitud,
    ids: np.ndarray,
    indices_ordenados: np.ndarray,
    similitudes: np.ndarray,
    ids_a_excluir: set,
    techo_similitud: float,
) -> Optional[Tuple[int, float]]:
    """
    Elige, entre los candidatos ya ordenados por similitud descendente, el
    mejor que caiga en la banda [UMBRAL_TFIDF_MAX_NEGATIVO, techo_similitud).
    El techo está atado al positivo real de ESTA tripleta
    (sim_positivo - margen), no a un valor fijo global, para garantizar
    sim(ancla,positivo) > sim(ancla,hard_negativo) siempre. El piso
    (UMBRAL_TFIDF_MAX_NEGATIVO, el mismo techo usado para negativos fáciles en
    la etapa 2) garantiza que siga siendo claramente más difícil que un
    negativo fácil. Prioriza género disjunto sobre mismo género como desempate.
    """
    candidatos = []
    for posicion in indices_ordenados:
        similitud = float(similitudes[posicion])
        if similitud >= techo_similitud:
            continue  # no cumple sim(A,N) < sim(A,P) - margen; puede haber candidatos más abajo
        if ids[posicion] in ids_a_excluir:
            continue
        if similitud < UMBRAL_TFIDF_MAX_NEGATIVO:
            break  # ordenados descendentemente: ya no hay candidatos "difíciles" por debajo
        jaccard_genero = _jaccard(estructuras.generos_sets[posicion_ancla], estructuras.generos_sets[posicion])
        candidatos.append((int(posicion), similitud, jaccard_genero))

    if not candidatos:
        return None

    candidatos.sort(key=lambda c: (c[2] == 0.0, c[1]), reverse=True)
    posicion, similitud, _ = candidatos[0]
    return posicion, similitud


def generar_hard_negativos(
    peliculas: pd.DataFrame,
    estructuras: EstructurasSimilitud,
    positivos: pd.DataFrame,
    margen: float = MARGEN_HARD_NEGATIVO,
) -> pd.DataFrame:
    """
    Para cada par (ancla, positivo) ya existente, mina UN hard negative que
    garantiza sim(ancla,positivo) > sim(ancla,hard_negativo) + margen, y que
    sigue siendo claramente más difícil que un negativo fácil (piso =
    UMBRAL_TFIDF_MAX_NEGATIVO, el mismo techo de la etapa 2 para negativos
    fáciles). El techo NO es un umbral global fijo: depende de qué tan
    parecido es el positivo real de esa tripleta puntual.
    """
    if positivos.empty:
        logger.warning("No hay pares positivos: no se puede hacer Hard Negative Mining.")
        return pd.DataFrame()

    ids = peliculas["id"].to_numpy()
    titulos = peliculas["titulo"].to_numpy()
    posicion_por_id = {pid: pos for pos, pid in enumerate(ids)}

    filas = []
    pares_sin_candidato = 0
    anclas_procesadas = 0
    total_anclas = positivos["id_ancla"].nunique()

    for id_ancla, grupo in positivos.groupby("id_ancla"):
        posicion_ancla = posicion_por_id.get(id_ancla)
        if posicion_ancla is None:
            continue

        anclas_procesadas += 1
        if anclas_procesadas % 2000 == 0:
            logger.info("Hard Negative Mining: %d/%d anclas procesadas...", anclas_procesadas, total_anclas)

        ids_a_excluir = {id_ancla} | set(grupo["id_positivo"])
        indices_ordenados, similitudes = _candidatos_ordenados_por_similitud(posicion_ancla, estructuras)

        for fila_positivo in grupo.itertuples(index=False):
            similitud_positivo = float(fila_positivo.similitud_tfidf)
            techo = min(similitud_positivo - margen, UMBRAL_HARD_NEG_SIM_MAX)

            if techo <= UMBRAL_TFIDF_MAX_NEGATIVO:
                pares_sin_candidato += 1
                continue  # el positivo no es lo bastante parecido como para dejar una banda válida

            candidato = _elegir_hard_negativo(
                posicion_ancla, estructuras, ids, indices_ordenados, similitudes, ids_a_excluir, techo
            )
            if candidato is None:
                pares_sin_candidato += 1
                continue

            posicion_candidato, similitud = candidato
            jaccard_genero = _jaccard(
                estructuras.generos_sets[posicion_ancla], estructuras.generos_sets[posicion_candidato]
            )
            tipo = "genero_distinto" if jaccard_genero == 0.0 else "mismo_genero_dificil"

            filas.append({
                "id_ancla": id_ancla,
                "titulo_ancla": titulos[posicion_ancla],
                "id_positivo": fila_positivo.id_positivo,
                "similitud_tfidf_positivo": round(similitud_positivo, 4),
                "id_negativo": ids[posicion_candidato],
                "titulo_negativo": titulos[posicion_candidato],
                "jaccard_genero": round(jaccard_genero, 4),
                "similitud_tfidf": round(similitud, 4),
                "tipo_negativo": tipo,
            })

    if pares_sin_candidato:
        logger.warning(
            "%d pares (ancla, positivo) sin hard negative válido (banda [%.2f, sim_positivo-%.2f) vacía o sin candidatos).",
            pares_sin_candidato, UMBRAL_TFIDF_MAX_NEGATIVO, margen,
        )

    resultado = pd.DataFrame(filas)
    logger.info("Hard negatives generados: %d (de %d pares positivos)", len(resultado), len(positivos))
    return resultado


# ===============================
# 5. TRIPLETAS MEJORADAS
# ===============================
def construir_tripletas_hard(
    peliculas: pd.DataFrame, positivos: pd.DataFrame, hard_negativos: pd.DataFrame
) -> pd.DataFrame:
    """
    Empareja cada positivo con el hard negative minado específicamente para
    ese par (ancla, positivo). La correspondencia ya es exacta 1:1 por
    construcción, así que se arma con un merge directo en vez de reutilizar
    el emparejador genérico de la etapa 2 (que cicla cuando las cantidades no
    coinciden; innecesario aquí).
    """
    if hard_negativos.empty:
        logger.warning("Sin hard negatives: no se generan tripletas hard.")
        return pd.DataFrame()

    texto_por_id = dict(zip(peliculas["id"], peliculas["texto"]))

    columnas_hard = hard_negativos[[
        "id_ancla", "id_positivo", "id_negativo", "titulo_negativo",
        "jaccard_genero", "similitud_tfidf", "tipo_negativo",
    ]].rename(columns={"jaccard_genero": "jaccard_genero_negativo", "similitud_tfidf": "similitud_tfidf_negativo"})

    tripletas_hard = positivos.merge(columnas_hard, on=["id_ancla", "id_positivo"], how="inner")

    tripletas_hard = tripletas_hard.rename(columns={
        "id_ancla": "anchor_id",
        "titulo_ancla": "anchor_titulo",
        "id_positivo": "positive_id",
        "titulo_positivo": "positive_titulo",
        "id_negativo": "negative_id",
        "titulo_negativo": "negative_titulo",
        "score": "score_positivo",
    })

    tripletas_hard["anchor_texto"] = tripletas_hard["anchor_id"].map(texto_por_id)
    tripletas_hard["positive_texto"] = tripletas_hard["positive_id"].map(texto_por_id)
    tripletas_hard["negative_texto"] = tripletas_hard["negative_id"].map(texto_por_id)

    logger.info("Tripletas hard construidas: %d", len(tripletas_hard))
    return tripletas_hard


def guardar_tripletas_hard_y_v2(
    tripletas_facil: pd.DataFrame,
    tripletas_hard: pd.DataFrame,
    ruta_tripletas_hard: Path = RUTA_TRIPLETAS_HARD,
    ruta_dataset_v2: Path = RUTA_DATASET_V2,
) -> None:
    """
    Guarda `tripletas_hard.csv` (vista legible: ids, títulos y métricas) y
    `dataset_entrenamiento_v2.csv` (anchor/positive/negative en texto,
    combinando las tripletas fáciles -ya con la cobertura mejorada- con las
    tripletas hard, marcadas con la columna 'dificultad'). Los archivos de la
    etapa 2 (pares_positivos.csv, tripletas.csv, dataset_entrenamiento.csv,
    etc.) no se tocan: estos son artefactos propios de esta etapa y se
    regeneran por completo en cada corrida.
    """
    ruta_tripletas_hard.parent.mkdir(parents=True, exist_ok=True)

    columnas_legibles = [
        "anchor_id", "anchor_titulo",
        "positive_id", "positive_titulo",
        "negative_id", "negative_titulo",
        "score_positivo", "similitud_tfidf_negativo", "tipo_negativo",
    ]
    if tripletas_hard.empty:
        pd.DataFrame(columns=columnas_legibles).to_csv(ruta_tripletas_hard, index=False)
    else:
        columnas_presentes = [c for c in columnas_legibles if c in tripletas_hard.columns]
        tripletas_hard[columnas_presentes].to_csv(ruta_tripletas_hard, index=False)
    logger.info("Tripletas hard guardadas en: %s", ruta_tripletas_hard)

    facil_v2 = tripletas_facil[["anchor_texto", "positive_texto", "negative_texto"]].rename(
        columns={"anchor_texto": "anchor", "positive_texto": "positive", "negative_texto": "negative"}
    )
    facil_v2["dificultad"] = "facil"

    if not tripletas_hard.empty:
        dificil_v2 = tripletas_hard[["anchor_texto", "positive_texto", "negative_texto"]].rename(
            columns={"anchor_texto": "anchor", "positive_texto": "positive", "negative_texto": "negative"}
        )
        dificil_v2["dificultad"] = "dificil"
        dataset_v2 = pd.concat([facil_v2, dificil_v2], ignore_index=True)
    else:
        dataset_v2 = facil_v2

    dataset_v2.to_csv(ruta_dataset_v2, index=False)
    logger.info("Dataset de entrenamiento v2 guardado en: %s (%d filas)", ruta_dataset_v2, len(dataset_v2))


# ===============================
# 6. EVALUACIÓN
# ===============================
def evaluar_mejora_tripletas(
    positivos: pd.DataFrame,
    tripletas_facil: pd.DataFrame,
    tripletas_hard: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compara tripletas fáciles vs. hard: cantidad y similitud TF-IDF promedio
    ancla-positivo (columna 'similitud_tfidf', ya calculada al generar los
    positivos: reutilizada, no recalculada) y ancla-negativo (fácil vs. hard).
    También reporta la similitud ancla-positivo restringida al subconjunto de
    pares que sí obtuvo un hard negative, para comparar ambas brechas sobre
    la misma base de pares.
    """
    similitud_positivo_general = float(positivos["similitud_tfidf"].mean()) if not positivos.empty else 0.0
    similitud_positivo_subset_hard = (
        float(tripletas_hard["similitud_tfidf"].mean()) if not tripletas_hard.empty else 0.0
    )

    evaluacion = {
        "cantidad_tripletas_originales": int(len(tripletas_facil)),
        "cantidad_tripletas_hard": int(len(tripletas_hard)),
        "similitud_promedio_anchor_positivo": similitud_positivo_general,
        "similitud_promedio_anchor_positivo_subset_hard": similitud_positivo_subset_hard,
        "similitud_promedio_anchor_negativo_facil": (
            float(tripletas_facil["similitud_tfidf_negativo"].mean()) if not tripletas_facil.empty else 0.0
        ),
        "similitud_promedio_anchor_negativo_hard": (
            float(tripletas_hard["similitud_tfidf_negativo"].mean()) if not tripletas_hard.empty else 0.0
        ),
    }

    brecha_facil = evaluacion["similitud_promedio_anchor_positivo"] - evaluacion["similitud_promedio_anchor_negativo_facil"]
    brecha_hard = evaluacion["similitud_promedio_anchor_positivo_subset_hard"] - evaluacion["similitud_promedio_anchor_negativo_hard"]
    evaluacion["brecha_positivo_negativo_facil"] = round(brecha_facil, 4)
    evaluacion["brecha_positivo_negativo_hard"] = round(brecha_hard, 4)

    logger.info(
        "Evaluación: sim(A,P)=%.4f | sim(A,N_facil)=%.4f (brecha %.4f) | "
        "sim(A,P)_subset_hard=%.4f | sim(A,N_hard)=%.4f (brecha %.4f)",
        evaluacion["similitud_promedio_anchor_positivo"],
        evaluacion["similitud_promedio_anchor_negativo_facil"],
        brecha_facil,
        similitud_positivo_subset_hard,
        evaluacion["similitud_promedio_anchor_negativo_hard"],
        brecha_hard,
    )
    return evaluacion


# ===============================
# ORQUESTACIÓN
# ===============================
def main() -> None:
    # Todas las rutas se leen aquí de las constantes del módulo (no de los
    # defaults de cada función) para que sobrescribir esas constantes -por
    # ejemplo en un test- se refleje realmente en la ejecución completa.
    try:
        peliculas, positivos, negativos, tripletas, dataset_entrenamiento = cargar_todos_los_datos(
            RUTA_PELICULAS, RUTA_POSITIVOS, RUTA_NEGATIVOS, RUTA_TRIPLETAS, RUTA_DATASET_ENTRENAMIENTO
        )
    except (FileNotFoundError, ValueError):
        logger.exception("No se pudo cargar el dataset de entrada.")
        raise

    # Estadísticas de la corrida anterior, leídas ANTES de sobrescribir el JSON.
    estadisticas_anteriores = cargar_estadisticas_anteriores(RUTA_ESTADISTICAS_JSON)

    estructuras = _construir_estructuras_similitud(peliculas)

    # --- Cobertura de positivos: fallback Top-1 para películas sin cobertura ---
    positivos_fallback = generar_positivos_fallback(peliculas, estructuras, positivos)
    positivos_mejorado = combinar_positivos(positivos, positivos_fallback)
    guardar_positivos_mejorado(positivos_mejorado, RUTA_POSITIVOS_MEJORADO)

    # Tripletas "fáciles" ampliadas: mismos negativos de la etapa 2 (ya cubren
    # el 100% de las películas), pero ahora emparejadas con la cobertura de
    # positivos ampliada.
    tripletas_mejoradas = generar_tripletas_base(peliculas, positivos_mejorado, negativos)

    # --- Parte 1: validación (sobre la cobertura mejorada) ---
    incidencias = ejecutar_validaciones(peliculas, positivos_mejorado, negativos, tripletas_mejoradas)

    # --- Parte 2: estadísticas ---
    estadisticas = calcular_estadisticas(peliculas, tripletas_mejoradas)
    estadisticas = agregar_resumen_validacion(estadisticas, incidencias)
    estadisticas["cobertura"] = calcular_cobertura(peliculas, positivos_mejorado, negativos, tripletas_mejoradas)

    # --- Parte 3: visualizaciones ---
    generar_graficos(peliculas, tripletas_mejoradas, estadisticas, DIR_GRAFICOS)

    # --- Parte 4 y 5: hard negative mining (margen atado al positivo real) + tripletas mejoradas ---
    hard_negativos = generar_hard_negativos(peliculas, estructuras, positivos_mejorado)
    tripletas_hard = construir_tripletas_hard(peliculas, positivos_mejorado, hard_negativos)
    guardar_tripletas_hard_y_v2(tripletas_mejoradas, tripletas_hard, RUTA_TRIPLETAS_HARD, RUTA_DATASET_V2)

    # --- Parte 6: evaluación + comparación con la corrida anterior ---
    evaluacion = evaluar_mejora_tripletas(positivos_mejorado, tripletas_mejoradas, tripletas_hard)
    estadisticas["evaluacion_hard_negative_mining"] = evaluacion
    comparacion = construir_comparacion(estadisticas_anteriores, estadisticas)

    # --- Reporte final ---
    reporte = construir_reporte_texto(estadisticas, incidencias, evaluacion, comparacion)
    guardar_reporte_texto(reporte, RUTA_REPORTE_TXT)
    guardar_estadisticas_json(estadisticas, RUTA_ESTADISTICAS_JSON)

    print("\n" + "=" * 60)
    print("ETAPA 3 (CORREGIDA) COMPLETADA: cobertura mejorada y hard negatives con margen garantizado.")
    print("(Todavía sin embeddings, sin ChromaDB, sin fine-tuning, sin cambios en FastAPI)")
    print("=" * 60)
    print("\n" + comparacion)


if __name__ == "__main__":
    main()
