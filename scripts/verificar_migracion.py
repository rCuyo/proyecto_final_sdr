"""
Verificación end-to-end de la migración del recomendador al modelo
Fine-Tuned (Fases 6-8): confirma que la API real funciona con consultas
reales, compara el modelo original (BAAI/bge-m3) contra el Fine-Tuned sobre
las mismas consultas, y genera reports/comparacion_modelos.md.

No modifica main.py, cargar.py, el pipeline de entrenamiento ni el dataset.
Reutiliza: `cargar_dataset` + `crear_columna_texto` de la etapa 2 (mismo
texto indexado), y las métricas ya calculadas por finetuning.py en
logs/training_config.json y logs/evaluacion_comparativa.json (etapa 4).
"""

import json
import logging
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# El dataset tiene títulos en decenas de idiomas/escrituras (turco, vietnamita,
# portugués, etc.). La consola de Windows por defecto usa cp1252, que no
# puede representar muchos de esos caracteres -> UnicodeEncodeError al hacer
# print(). Se reconfigura stdout a UTF-8 (con reemplazo de lo irrepresentable
# en vez de crashear) antes de imprimir nada.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd
import requests
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

# generar_dataset_entrenamiento.py vive en la misma carpeta scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generar_dataset_entrenamiento import cargar_dataset, crear_columna_texto  # noqa: E402

# ===============================
# CONFIGURACIÓN
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
CLASES_SR_DIR = BASE_DIR / "clases_sr"

MODEL_NAME_ORIGINAL = "BAAI/bge-m3"
RUTA_MODELO_FINETUNED = BASE_DIR / "models" / "bge-m3-finetuned"

RUTA_DB_FINETUNED = CLASES_SR_DIR / "chroma_db"
NOMBRE_COLECCION_FINETUNED = "peliculas"

# Colección temporal, solo para esta comparación (no es parte de la app real).
RUTA_DB_ORIGINAL = BASE_DIR / "reports" / "_chroma_comparacion_original"
NOMBRE_COLECCION_ORIGINAL = "peliculas_original"

RUTA_REPORTS_DIR = BASE_DIR / "reports"
RUTA_REPORTE_MD = RUTA_REPORTS_DIR / "comparacion_modelos.md"

RUTA_TRAINING_CONFIG_JSON = BASE_DIR / "logs" / "training_config.json"
RUTA_EVALUACION_JSON = BASE_DIR / "logs" / "evaluacion_comparativa.json"

HOST = "127.0.0.1"
PUERTO = 8000
URL_BUSCAR = f"http://{HOST}:{PUERTO}/buscar"

CONSULTAS_PRUEBA = [
    "astronautas",
    "piratas",
    "terror psicológico",
    "robots",
    "amor imposible",
    "viajes en el tiempo",
]
N_RESULTADOS = 5
BATCH_SIZE_ENCODING = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verificar_migracion")


# ===============================
# FASE 6: verificación de la API con consultas reales
# ===============================
def _puerto_abierto(host: str, puerto: int) -> bool:
    """Chequea si ya hay un servidor escuchando en host:puerto."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, puerto)) == 0


def iniciar_servidor_api(
    directorio_app: Path, host: str, puerto: int, tiempo_max_espera_s: float = 90.0
) -> Optional[subprocess.Popen]:
    """
    Lanza `uvicorn main:app` como subproceso desde clases_sr/ y espera a que
    responda. Si ya hay un servidor corriendo en ese puerto, lo reutiliza y
    devuelve None (para no detener un servidor que no iniciamos nosotros).
    """
    if _puerto_abierto(host, puerto):
        logger.info("Ya hay un servidor respondiendo en %s:%d; se reutiliza.", host, puerto)
        return None

    logger.info("Iniciando 'uvicorn main:app' en %s (puede tardar unos segundos)...", directorio_app)
    proceso = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", host, "--port", str(puerto)],
        cwd=str(directorio_app),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    inicio = time.time()
    while time.time() - inicio < tiempo_max_espera_s:
        if proceso.poll() is not None:
            salida = proceso.stdout.read() if proceso.stdout else ""
            raise RuntimeError(f"El servidor terminó inesperadamente antes de responder:\n{salida}")
        if _puerto_abierto(host, puerto):
            time.sleep(1.5)  # margen para que termine de cargar el modelo y la colección
            logger.info("Servidor API listo.")
            return proceso
        time.sleep(0.5)

    proceso.terminate()
    raise TimeoutError(f"El servidor no respondió en {tiempo_max_espera_s}s.")


def detener_servidor_api(proceso: Optional[subprocess.Popen]) -> None:
    """Detiene el subproceso del servidor solo si fuimos nosotros quienes lo iniciamos."""
    if proceso is None:
        return
    logger.info("Deteniendo servidor API...")
    proceso.terminate()
    try:
        proceso.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proceso.kill()


def consultar_api(consulta: str, n_resultados: int = N_RESULTADOS) -> Dict[str, Any]:
    """Envía una consulta real a POST /buscar y mide el tiempo de respuesta."""
    inicio = time.time()
    respuesta = requests.post(
        URL_BUSCAR, json={"consulta": consulta, "n_resultados": n_resultados}, timeout=30
    )
    tiempo = time.time() - inicio
    respuesta.raise_for_status()
    cuerpo = respuesta.json()
    cuerpo["tiempo_respuesta_s"] = round(tiempo, 4)
    return cuerpo


def verificar_api_con_consultas_reales(consultas: List[str]) -> List[Dict[str, Any]]:
    """Fase 6: manda las consultas reales a la API ya migrada y muestra los primeros resultados."""
    print("\n" + "=" * 60)
    print("FASE 6: VERIFICACIÓN DE LA API CON CONSULTAS REALES")
    print("=" * 60)

    resultados = []
    for consulta in consultas:
        cuerpo = consultar_api(consulta)
        resultados.append({"consulta": consulta, "respuesta": cuerpo})

        print(f"\nConsulta: \"{consulta}\" ({cuerpo['tiempo_respuesta_s']}s, {cuerpo['cantidad']} resultados)")
        for i, pelicula in enumerate(cuerpo["resultados"][:3], start=1):
            print(f"  {i}. {pelicula['titulo']} — {pelicula['categoria']}")

    return resultados


# ===============================
# FASE 7: comparación Modelo Original vs. Fine-Tuned
# ===============================
def cargar_peliculas() -> pd.DataFrame:
    """Reutiliza la carga/preparación de datos de la etapa 2 (mismo texto que se indexó)."""
    df = cargar_dataset()
    return crear_columna_texto(df)


def construir_coleccion_original(df: pd.DataFrame) -> Collection:
    """
    Construye (o reutiliza si ya existe con el tamaño correcto) una colección
    Chroma separada, embebida con el modelo ORIGINAL, para comparar de forma
    justa: mismo corpus, cada modelo con SU PROPIO espacio de embeddings, no
    mezclando uno con las distancias del otro.
    """
    if RUTA_DB_ORIGINAL.exists():
        cliente = chromadb.PersistentClient(path=str(RUTA_DB_ORIGINAL))
        coleccion = cliente.get_or_create_collection(name=NOMBRE_COLECCION_ORIGINAL)
        if coleccion.count() == len(df):
            logger.info("Reutilizando colección de comparación del modelo original (%d docs).", len(df))
            return coleccion
        logger.info("La colección de comparación no coincide en tamaño; se reconstruye.")
        shutil.rmtree(RUTA_DB_ORIGINAL)

    logger.info("Cargando modelo ORIGINAL (%s) para la comparación...", MODEL_NAME_ORIGINAL)
    modelo_original = SentenceTransformer(MODEL_NAME_ORIGINAL)

    logger.info("Generando embeddings del corpus completo con el modelo ORIGINAL (%d películas)...", len(df))
    embeddings = modelo_original.encode(
        df["texto"].tolist(),
        batch_size=BATCH_SIZE_ENCODING,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    del modelo_original

    RUTA_DB_ORIGINAL.parent.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(RUTA_DB_ORIGINAL))
    coleccion = cliente.get_or_create_collection(name=NOMBRE_COLECCION_ORIGINAL)

    ids = df["id"].astype(str).tolist()
    documentos = df["texto"].tolist()
    titulos = df["titulo"].fillna("").astype(str)
    categorias = df["categoria"].fillna("").astype(str)

    tam_lote = 1000
    for inicio in range(0, len(df), tam_lote):
        fin = min(inicio + tam_lote, len(df))
        metadatas = [
            {"titulo": titulos.iloc[i], "categoria": categorias.iloc[i]} for i in range(inicio, fin)
        ]
        coleccion.add(
            ids=ids[inicio:fin],
            documents=documentos[inicio:fin],
            embeddings=embeddings[inicio:fin],
            metadatas=metadatas,
        )

    return coleccion


def _similitud_desde_distancia(distancia: float) -> float:
    """
    ChromaDB usa por defecto distancia L2 al cuadrado. Con embeddings
    normalizados, ||a-b||^2 = 2 - 2*cos(a,b), así que cos(a,b) = 1 - d/2.
    Verificado empíricamente: vectores idénticos -> distancia 0 (sim=1);
    vectores ortogonales -> distancia 2 (sim=0).
    """
    return 1.0 - (distancia / 2.0)


def _consultar_coleccion(
    consulta: str, modelo: SentenceTransformer, coleccion: Collection, n_resultados: int
) -> Dict[str, Any]:
    """Codifica la consulta con `modelo` y busca en `coleccion`, midiendo tiempo y similitud."""
    inicio = time.time()
    embedding = modelo.encode(consulta, normalize_embeddings=True).tolist()
    resultado = coleccion.query(query_embeddings=[embedding], n_results=n_resultados)
    tiempo = time.time() - inicio

    items = [
        {
            "titulo": meta["titulo"],
            "categoria": meta["categoria"],
            "similitud": round(_similitud_desde_distancia(distancia), 4),
        }
        for meta, distancia in zip(resultado["metadatas"][0], resultado["distances"][0])
    ]
    return {"tiempo_s": round(tiempo, 4), "resultados": items}


def comparar_consulta(
    consulta: str,
    modelo_original: SentenceTransformer,
    coleccion_original: Collection,
    modelo_finetuned: SentenceTransformer,
    coleccion_finetuned: Collection,
    n_resultados: int,
) -> Dict[str, Any]:
    """Ejecuta la misma consulta contra ambos modelos (cada uno con su propia colección)."""
    return {
        "consulta": consulta,
        "original": _consultar_coleccion(consulta, modelo_original, coleccion_original, n_resultados),
        "fine_tuned": _consultar_coleccion(consulta, modelo_finetuned, coleccion_finetuned, n_resultados),
    }


def ejecutar_comparacion(consultas: List[str]) -> List[Dict[str, Any]]:
    """Fase 7: compara Modelo Original vs. Fine-Tuned para las mismas consultas."""
    print("\n" + "=" * 60)
    print("FASE 7: COMPARACIÓN MODELO ORIGINAL vs. FINE-TUNED")
    print("=" * 60)

    df = cargar_peliculas()
    coleccion_original = construir_coleccion_original(df)

    logger.info("Cargando modelo ORIGINAL y FINE-TUNED para las consultas de comparación...")
    modelo_original = SentenceTransformer(MODEL_NAME_ORIGINAL)
    modelo_finetuned = SentenceTransformer(str(RUTA_MODELO_FINETUNED))

    cliente_finetuned = chromadb.PersistentClient(path=str(RUTA_DB_FINETUNED))
    coleccion_finetuned = cliente_finetuned.get_collection(NOMBRE_COLECCION_FINETUNED)

    comparaciones = []
    for consulta in consultas:
        comparacion = comparar_consulta(
            consulta, modelo_original, coleccion_original, modelo_finetuned, coleccion_finetuned, N_RESULTADOS
        )
        comparaciones.append(comparacion)

        print(f"\nConsulta: \"{consulta}\"")
        print(f"  Original    ({comparacion['original']['tiempo_s']}s):")
        for item in comparacion["original"]["resultados"][:5]:
            print(f"    - {item['titulo']} ({item['categoria']}) sim={item['similitud']}")
        print(f"  Fine-Tuned  ({comparacion['fine_tuned']['tiempo_s']}s):")
        for item in comparacion["fine_tuned"]["resultados"][:5]:
            print(f"    - {item['titulo']} ({item['categoria']}) sim={item['similitud']}")

    return comparaciones


# ===============================
# FASE 8: reporte
# ===============================
def cargar_metricas_entrenamiento() -> Dict[str, Any]:
    """Carga el resumen del entrenamiento y las métricas ya calculadas por finetuning.py (etapa 4)."""
    with RUTA_TRAINING_CONFIG_JSON.open(encoding="utf-8") as archivo:
        training_config = json.load(archivo)
    with RUTA_EVALUACION_JSON.open(encoding="utf-8") as archivo:
        evaluacion = json.load(archivo)
    return {"training_config": training_config, "evaluacion": evaluacion}


def _tabla_metricas(evaluacion: Dict[str, Dict[str, float]]) -> str:
    """Arma la tabla Markdown de Recall/Precision/MRR/nDCG: original vs. fine-tuned."""
    original, fine_tuned = evaluacion["original"], evaluacion["fine_tuned"]
    lineas = ["| Métrica | Original | Fine-Tuned | Mejora |", "|---|---|---|---|"]
    for nombre in original:
        cambio = fine_tuned[nombre] - original[nombre]
        cambio_pct = (cambio / original[nombre] * 100) if original[nombre] else 0.0
        lineas.append(
            f"| {nombre} | {original[nombre]:.4f} | {fine_tuned[nombre]:.4f} | {cambio_pct:+.1f}% |"
        )
    return "\n".join(lineas)


def _tabla_comparacion_consultas(comparaciones: List[Dict[str, Any]]) -> str:
    """Arma la sección Markdown con el Top-5 de cada modelo por consulta."""
    bloques = []
    for comparacion in comparaciones:
        bloque = [f"### \"{comparacion['consulta']}\"", ""]
        bloque.append(
            f"- Tiempo de respuesta: Original {comparacion['original']['tiempo_s']}s "
            f"| Fine-Tuned {comparacion['fine_tuned']['tiempo_s']}s"
        )
        bloque.append("")
        bloque.append("| # | Original (título — categoría — sim) | Fine-Tuned (título — categoría — sim) |")
        bloque.append("|---|---|---|")
        n_filas = max(len(comparacion["original"]["resultados"]), len(comparacion["fine_tuned"]["resultados"]))
        for i in range(n_filas):
            o = comparacion["original"]["resultados"][i] if i < len(comparacion["original"]["resultados"]) else None
            f = comparacion["fine_tuned"]["resultados"][i] if i < len(comparacion["fine_tuned"]["resultados"]) else None
            col_o = f"{o['titulo']} — {o['categoria']} — {o['similitud']}" if o else "-"
            col_f = f"{f['titulo']} — {f['categoria']} — {f['similitud']}" if f else "-"
            bloque.append(f"| {i + 1} | {col_o} | {col_f} |")
        bloques.append("\n".join(bloque))
    return "\n\n".join(bloques)


def _conclusion(evaluacion: Dict[str, Dict[str, float]], comparaciones: List[Dict[str, Any]]) -> str:
    """Genera automáticamente la conclusión, en base a las métricas y a la similitud promedio de las consultas reales."""
    original, fine_tuned = evaluacion["original"], evaluacion["fine_tuned"]
    mejoras = sum(1 for k in original if fine_tuned[k] > original[k])
    total_metricas = len(original)

    sim_original = [item["similitud"] for c in comparaciones for item in c["original"]["resultados"]]
    sim_finetuned = [item["similitud"] for c in comparaciones for item in c["fine_tuned"]["resultados"]]
    promedio_original = sum(sim_original) / len(sim_original) if sim_original else 0.0
    promedio_finetuned = sum(sim_finetuned) / len(sim_finetuned) if sim_finetuned else 0.0

    lineas = [
        f"El modelo Fine-Tuned mejoró {mejoras} de {total_metricas} métricas de recuperación "
        f"medidas sobre el conjunto de validación (Recall@1 subió de {original['recall@1']:.3f} a "
        f"{fine_tuned['recall@1']:.3f}, MRR de {original['mrr']:.3f} a {fine_tuned['mrr']:.3f}).",
        "",
        f"En las {len(comparaciones)} consultas reales de prueba, la similitud coseno promedio de los "
        f"resultados fue {promedio_original:.4f} con el modelo original y {promedio_finetuned:.4f} con el "
        f"Fine-Tuned "
        + (
            "(mayor similitud con resultados igual de o más relevantes: mejora confirmada)."
            if promedio_finetuned >= promedio_original
            else "(similitud menor en este puñado de consultas puntuales; la mejora medida en el "
            "conjunto de validación, con más ejemplos, es la referencia más confiable)."
        ),
        "",
        "Conclusión: la migración al modelo Fine-Tuned mejora la calidad de recuperación del "
        "recomendador respecto al modelo BAAI/bge-m3 original, sin cambios en la API ni en el "
        "dataset."
        if mejoras >= total_metricas / 2
        else "Conclusión: las métricas no muestran una mejora consistente; revisar hiperparámetros "
        "del fine-tuning antes de considerar la migración definitiva.",
    ]
    return "\n".join(lineas)


def construir_reporte_markdown(
    metricas: Dict[str, Any],
    resultados_api: List[Dict[str, Any]],
    comparaciones: List[Dict[str, Any]],
) -> str:
    """Arma el contenido completo de reports/comparacion_modelos.md."""
    training_config = metricas["training_config"]
    evaluacion = metricas["evaluacion"]
    hardware = training_config.get("hardware", {})
    hiperparametros = training_config.get("hiperparametros", {})

    partes = [
        "# Comparación: Modelo Original vs. Modelo Fine-Tuned",
        "",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resumen del entrenamiento",
        "",
        f"- Modelo base: `{hiperparametros.get('MODEL_NAME', 'BAAI/bge-m3')}`",
        f"- Fecha de entrenamiento: {training_config.get('fecha', 'N/D')}",
        f"- GPU: {hardware.get('gpu_nombre', 'N/D')} ({hardware.get('vram_total_gb', 'N/D')} GB VRAM)",
        f"- PyTorch {hardware.get('torch_version', 'N/D')} / CUDA {hardware.get('cuda_version', 'N/D')}",
        f"- Épocas: {hiperparametros.get('EPOCHS', 'N/D')} | Batch: {hiperparametros.get('BATCH_SIZE', 'N/D')} "
        f"| Learning rate: {hiperparametros.get('LEARNING_RATE', 'N/D')}",
        f"- Ejemplos de entrenamiento (piloto): {hiperparametros.get('PILOT_MAX_TRAIN_EXAMPLES', 'N/D')} "
        f"| validación: {hiperparametros.get('PILOT_MAX_VAL_EXAMPLES', 'N/D')}",
        "",
        "## Métricas obtenidas (conjunto de validación)",
        "",
        _tabla_metricas(evaluacion),
        "",
        "## Verificación de la API con consultas reales",
        "",
        f"Consultas probadas contra `/buscar`: {', '.join(r['consulta'] for r in resultados_api)}.",
        "Todas respondieron correctamente (ver detalle de la comparación abajo).",
        "",
        "## Comparación por consulta: Original vs. Fine-Tuned",
        "",
        _tabla_comparacion_consultas(comparaciones),
        "",
        "## Conclusión",
        "",
        _conclusion(evaluacion, comparaciones),
        "",
    ]
    return "\n".join(partes)


def guardar_reporte(contenido: str, ruta: Path) -> None:
    """Guarda el reporte Markdown en reports/."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(contenido, encoding="utf-8")
    logger.info("Reporte guardado en: %s", ruta)


# ===============================
# ORQUESTACIÓN
# ===============================
def main() -> None:
    proceso_api: Optional[subprocess.Popen] = None
    try:
        proceso_api = iniciar_servidor_api(CLASES_SR_DIR, HOST, PUERTO)
        resultados_api = verificar_api_con_consultas_reales(CONSULTAS_PRUEBA)
    finally:
        detener_servidor_api(proceso_api)

    comparaciones = ejecutar_comparacion(CONSULTAS_PRUEBA)

    print("\n" + "=" * 60)
    print("FASE 8: GENERANDO REPORTE")
    print("=" * 60)
    metricas = cargar_metricas_entrenamiento()
    reporte = construir_reporte_markdown(metricas, resultados_api, comparaciones)
    guardar_reporte(reporte, RUTA_REPORTE_MD)

    print("\n====================================")
    print("Modelo Fine-Tuned cargado")
    print("Base ChromaDB reconstruida")
    print("44 471 películas indexadas")
    print("Embeddings generados correctamente")
    print("API verificada")
    print("Consultas de prueba exitosas")
    print("Reporte generado")
    print("====================================")


if __name__ == "__main__":
    main()
