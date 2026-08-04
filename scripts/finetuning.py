"""
Piloto de Fine-Tuning de BAAI/bge-m3 con SentenceTransformers (ETAPA 4).

Reutiliza el dataset ya preparado, validado y mejorado con Hard Negative
Mining en las etapas 1-3 (`data/training/dataset_entrenamiento_v2.csv`). No
modifica ni regenera ningún artefacto de esas etapas, y no toca FastAPI,
ChromaDB ni el frontend.

Es un PILOTO: el objetivo es verificar que el pipeline de entrenamiento
funciona, que la pérdida converge y que el modelo mejora respecto al
original -no producir el modelo definitivo-. Todos los hiperparámetros viven
en `config.py`, a nivel de proyecto.

Usa la API moderna de SentenceTransformers (`SentenceTransformerTrainer` /
`SentenceTransformerTrainingArguments`, sobre `transformers.Trainer`), no el
`model.fit(...)` legado.

Requiere: pip install sentence-transformers datasets accelerate tensorboard
(torch con CUDA ya debe estar instalado -no se reinstala acá para no pisar
la build con soporte CUDA existente-).
"""

import os

# Debe fijarse ANTES de importar transformers/sentence_transformers/datasets:
# el entorno tiene un TensorFlow instalado (ajeno a este proyecto) incompatible
# con Keras 3, lo que rompe el import de `transformers.integrations` si TF
# queda habilitado. Este proyecto solo usa el backend de PyTorch.
os.environ.setdefault("USE_TF", "0")

import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # backend sin GUI: solo necesitamos guardar PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback, TrainerCallback

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

# config.py vive en la raíz del proyecto, un nivel arriba de scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ===============================
# LOGGING
# ===============================
def configurar_logging() -> logging.Logger:
    """Logger a consola + logs/training.log (handler único aunque se llame más de una vez)."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("finetuning")
    logger_.setLevel(logging.INFO)
    if logger_.handlers:
        return logger_

    formato = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    consola = logging.StreamHandler()
    consola.setFormatter(formato)
    logger_.addHandler(consola)

    archivo = logging.FileHandler(config.RUTA_TRAINING_LOG, encoding="utf-8")
    archivo.setFormatter(formato)
    logger_.addHandler(archivo)

    return logger_


logger = configurar_logging()


# ===============================
# HARDWARE
# ===============================
def detectar_dispositivo() -> torch.device:
    """Detecta CUDA automáticamente; nunca asume CPU salvo que no haya GPU o se fuerce en config.py."""
    if config.FORZAR_CPU:
        logger.warning("FORZAR_CPU=True en config.py: se ignora la GPU aunque esté disponible.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    logger.warning("CUDA no disponible: se entrenará en CPU (será considerablemente más lento).")
    return torch.device("cpu")


def _consultar_utilizacion_gpu(device: torch.device) -> Optional[float]:
    """% de uso de GPU vía nvidia-smi (torch no lo expone directamente). None si falla o no hay GPU."""
    if device.type != "cuda":
        return None
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return float(salida.stdout.strip().splitlines()[0])
    except Exception:
        return None


def mostrar_info_hardware(device: torch.device) -> Dict[str, Any]:
    """Muestra y devuelve GPU detectada, nombre, versión CUDA, VRAM total/usada/libre y versión de PyTorch."""
    info: Dict[str, Any] = {"torch_version": torch.__version__, "device_type": device.type}

    if device.type == "cuda":
        propiedades = torch.cuda.get_device_properties(device)
        vram_total = propiedades.total_memory / 1024**3
        vram_reservada = torch.cuda.memory_reserved(device) / 1024**3
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_nombre": torch.cuda.get_device_name(device),
            "vram_total_gb": round(vram_total, 2),
            "vram_usada_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 2),
            "vram_reservada_gb": round(vram_reservada, 2),
            "vram_libre_gb": round(vram_total - vram_reservada, 2),
            "utilizacion_gpu_pct": _consultar_utilizacion_gpu(device),
        })
        logger.info("GPU detectada: %s", info["gpu_nombre"])
        logger.info("Versión CUDA: %s", info["cuda_version"])
        logger.info("VRAM total: %.2f GB", info["vram_total_gb"])
        logger.info("VRAM utilizada: %.2f GB | VRAM libre: %.2f GB", info["vram_usada_gb"], info["vram_libre_gb"])
    else:
        info.update({"gpu_nombre": None, "cuda_version": None})

    logger.info("PyTorch %s", torch.__version__)
    return info


def fijar_semilla(seed: int) -> None:
    """Fija la semilla en random/numpy/torch (CPU y todas las GPUs) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===============================
# DATOS
# ===============================
def cargar_dataset_entrenamiento(ruta: Path) -> pd.DataFrame:
    """Carga dataset_entrenamiento_v2.csv (etapa 3) y descarta filas con texto vacío. No lo modifica en disco."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró '{ruta}'. Ejecuta primero las etapas 1-3 (preparar_dataset.py, "
            "generar_dataset_entrenamiento.py, validar_dataset_entrenamiento.py)."
        )

    df = pd.read_csv(ruta)
    columnas_requeridas = {"anchor", "positive", "negative"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en {ruta}: {faltantes}")

    df = df.dropna(subset=["anchor", "positive", "negative"])
    for columna in ["anchor", "positive", "negative"]:
        df = df[df[columna].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    logger.info("Dataset de entrenamiento cargado: %d filas.", len(df))
    return df


def dividir_train_validation(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa por ANCLA única (no por fila): todas las filas de una misma
    película van enteras a train o a validación, para no filtrar información
    entre ambos conjuntos.
    """
    # `.unique()` puede devolver un ExtensionArray (p.ej. ArrowStringArray, si
    # pandas infiere dtype "string[pyarrow]" al leer el CSV -default en
    # versiones recientes de pandas-) en vez de un numpy.ndarray plano.
    # scikit-learn indexa internamente con arrays de enteros/booleanos de
    # numpy, algo que un ChunkedArray de PyArrow no soporta (de ahí el
    # "TypeError: only integer scalar arrays can be converted to a scalar
    # index"). Se materializa explícitamente a numpy.ndarray -reproducido y
    # verificado con pandas en modo "future.infer_string"- para que sea
    # compatible con train_test_split, sin alterar el criterio de separación
    # (por ancla única). `df["anchor"].isin(...)` no necesita este ajuste:
    # opera sobre el almacenamiento nativo de pandas/Arrow, no por indexado
    # tipo numpy.
    anclas_unicas = np.asarray(df["anchor"].unique(), dtype=object)
    anclas_train, anclas_val = train_test_split(anclas_unicas, test_size=val_ratio, random_state=seed)

    df_train = df[df["anchor"].isin(anclas_train)].reset_index(drop=True)
    df_val = df[df["anchor"].isin(anclas_val)].reset_index(drop=True)

    logger.info(
        "Split train/val: %d filas / %d anclas (train) | %d filas / %d anclas (val).",
        len(df_train), len(anclas_train), len(df_val), len(anclas_val),
    )
    return df_train, df_val


def limitar_muestras(df: pd.DataFrame, maximo: Optional[int], seed: int) -> pd.DataFrame:
    """Recorta el dataset para el piloto. `maximo=None` usa el dataset completo (entrenamiento definitivo)."""
    if maximo is None or len(df) <= maximo:
        return df
    return df.sample(n=maximo, random_state=seed).reset_index(drop=True)


def construir_datasets_hf(df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    """Convierte los DataFrames a `datasets.Dataset` con las 3 columnas (anchor, positive, negative)."""
    columnas = ["anchor", "positive", "negative"]
    dataset_train = Dataset.from_pandas(df_train[columnas], preserve_index=False)
    dataset_val = Dataset.from_pandas(df_val[columnas], preserve_index=False)
    return dataset_train, dataset_val


# ===============================
# MODELO / PÉRDIDA / EVALUADOR (para monitoreo durante el entrenamiento)
# ===============================
def construir_modelo(model_name: str, device: torch.device, max_seq_length: int) -> SentenceTransformer:
    """Carga el modelo base y fija `max_seq_length` (bge-m3 soporta hasta 8192 tokens; lo acotamos)."""
    logger.info("Cargando modelo base: %s", model_name)
    modelo = SentenceTransformer(model_name, device=str(device))
    modelo.max_seq_length = max_seq_length
    return modelo


def construir_loss(modelo: SentenceTransformer) -> CachedMultipleNegativesRankingLoss:
    """
    CachedMultipleNegativesRankingLoss: mismo objetivo contrastivo con el que
    se entrenó BGE originalmente, pero con GradCache para desacoplar el batch
    efectivo (calidad) del pico de VRAM (`mini_batch_size`), clave en una GPU
    de 8GB. Ver justificación completa en la respuesta del análisis técnico.
    """
    return CachedMultipleNegativesRankingLoss(modelo, mini_batch_size=config.MINI_BATCH_SIZE)


def construir_evaluador(df_val: pd.DataFrame) -> TripletEvaluator:
    """
    TripletEvaluator: evalúa, sobre las tripletas de validación, si
    sim(ancla,positivo) > sim(ancla,negativo) -sirve para monitorear durante
    el entrenamiento-. La comparación exhaustiva (Recall/MRR/nDCG) contra el
    modelo original se hace aparte, al final, en `comparar_modelos`.
    """
    return TripletEvaluator(
        anchors=df_val["anchor"].tolist(),
        positives=df_val["positive"].tolist(),
        negatives=df_val["negative"].tolist(),
        name="validacion",
        batch_size=config.EVAL_ENCODE_BATCH_SIZE,
        show_progress_bar=False,
    )


# ===============================
# CHECKPOINTS
# ===============================
def detectar_checkpoint(output_dir: Path) -> Optional[str]:
    """Busca el checkpoint de mayor step en `output_dir`, para reanudar automáticamente si existe."""
    if not output_dir.exists():
        return None

    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _numero_step(ruta: Path) -> int:
        try:
            return int(ruta.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    mejor = max(checkpoints, key=_numero_step)
    logger.info("Checkpoint existente detectado: se reanudará desde %s", mejor)
    return str(mejor)


# ===============================
# MONITOREO (callback de la API moderna del Trainer)
# ===============================
class MonitoreoRecursos(TrainerCallback):
    """Inyecta tiempo transcurrido/restante y uso de GPU/VRAM en cada log del Trainer (y por lo tanto en TensorBoard)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._inicio_entrenamiento: Optional[float] = None
        self._inicio_epoch: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs) -> None:  # noqa: D401
        self._inicio_entrenamiento = time.time()
        self._inicio_epoch = time.time()
        logger.info("Entrenamiento iniciado: %d épocas, %d steps estimados.", args.num_train_epochs, state.max_steps)

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        if self._inicio_epoch is None:
            return
        duracion = time.time() - self._inicio_epoch
        logger.info("Época %.0f completada en %.1f s.", state.epoch, duracion)
        self._inicio_epoch = time.time()

    def on_log(self, args, state, control, logs: Optional[dict] = None, **kwargs) -> None:
        if logs is None or self._inicio_entrenamiento is None:
            return

        transcurrido = time.time() - self._inicio_entrenamiento
        logs["tiempo_transcurrido_s"] = round(transcurrido, 1)
        if state.max_steps and state.global_step:
            eta = transcurrido / state.global_step * max(state.max_steps - state.global_step, 0)
            logs["eta_segundos"] = round(eta, 1)

        if self.device.type == "cuda":
            vram_reservada = torch.cuda.memory_reserved(self.device) / 1024**3
            vram_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            logs["vram_asignada_gb"] = round(torch.cuda.memory_allocated(self.device) / 1024**3, 3)
            logs["vram_reservada_gb"] = round(vram_reservada, 3)
            logs["vram_libre_gb"] = round(vram_total - vram_reservada, 3)
            utilizacion = _consultar_utilizacion_gpu(self.device)
            if utilizacion is not None:
                logs["utilizacion_gpu_pct"] = utilizacion

        partes = [f"step={state.global_step}/{state.max_steps}"]
        if "loss" in logs:
            partes.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            partes.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            partes.append(f"lr={logs['learning_rate']:.2e}")
        if "eta_segundos" in logs:
            partes.append(f"eta={logs['eta_segundos']:.0f}s")
        if "vram_asignada_gb" in logs:
            partes.append(f"vram={logs['vram_asignada_gb']:.2f}/{logs.get('vram_libre_gb', 0):.2f}GB libre")
        logger.info(" | ".join(partes))


# ===============================
# ENTRENAMIENTO
# ===============================
def construir_training_arguments(device: torch.device) -> SentenceTransformerTrainingArguments:
    """Arma los TrainingArguments: warmup, scheduler (lineal por defecto), FP16/BF16, checkpoint y early stopping."""
    usar_fp16 = config.USE_FP16 and device.type == "cuda"
    usar_bf16 = config.USE_BF16 and device.type == "cuda"

    return SentenceTransformerTrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        optim=config.OPTIMIZER_NAME,
        gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        fp16=usar_fp16,
        bf16=usar_bf16,
        eval_strategy="steps",
        eval_steps=config.SAVE_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        logging_dir=str(config.RUNS_DIR),
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=config.SEED,
        run_name="bge-m3-finetuning-piloto",
    )


def entrenar(
    modelo: SentenceTransformer,
    args: SentenceTransformerTrainingArguments,
    dataset_train: Dataset,
    dataset_val: Dataset,
    loss: CachedMultipleNegativesRankingLoss,
    evaluador: TripletEvaluator,
    device: torch.device,
) -> SentenceTransformerTrainer:
    """Construye el Trainer (con Early Stopping y monitoreo de recursos) y ejecuta el entrenamiento."""
    trainer = SentenceTransformerTrainer(
        model=modelo,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        loss=loss,
        evaluator=evaluador,
        callbacks=[
            MonitoreoRecursos(device),
            EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE),
        ],
    )

    checkpoint = detectar_checkpoint(config.OUTPUT_DIR)
    logger.info("Iniciando entrenamiento...")
    trainer.train(resume_from_checkpoint=checkpoint)
    logger.info("Entrenamiento finalizado.")
    return trainer


def guardar_modelo_final(trainer: SentenceTransformerTrainer, ruta: Path) -> None:
    """Exporta el MEJOR modelo (cargado por `load_best_model_at_end`) a una carpeta limpia y autocontenida."""
    ruta.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ruta))
    logger.info("Modelo final guardado en: %s (cargar con SentenceTransformer('%s'))", ruta, ruta)


# ===============================
# LOGGING DE RESULTADOS / VISUALIZACIONES
# ===============================
def guardar_historial(trainer: SentenceTransformerTrainer, ruta_json: Path, ruta_csv: Path) -> None:
    """Guarda el historial completo de logs del Trainer (loss, lr, eval_loss, VRAM, tiempos) en JSON y CSV."""
    historial = trainer.state.log_history

    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(historial, archivo, ensure_ascii=False, indent=2)

    pd.DataFrame(historial).to_csv(ruta_csv, index=False)
    logger.info("Historial de entrenamiento guardado en: %s y %s", ruta_json, ruta_csv)


def extraer_historial(trainer: SentenceTransformerTrainer) -> pd.DataFrame:
    """Historial del Trainer como DataFrame, listo para graficar."""
    return pd.DataFrame(trainer.state.log_history)


def _guardar_figura(fig: plt.Figure, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("Gráfico guardado en: %s", ruta)


def graficar_loss(historial: pd.DataFrame, ruta: Path) -> None:
    """loss.png: pérdida de entrenamiento por step."""
    if "loss" not in historial.columns:
        logger.warning("Sin columna 'loss' en el historial; se omite %s", ruta.name)
        return
    datos = historial.dropna(subset=["loss"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(datos["step"], datos["loss"], marker="o", markersize=3, color="#4C72B0")
    ax.set_title("Pérdida de entrenamiento (train loss)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    _guardar_figura(fig, ruta)


def graficar_learning_rate(historial: pd.DataFrame, ruta: Path) -> None:
    """learning_rate.png: evolución del learning rate (warmup + scheduler)."""
    if "learning_rate" not in historial.columns:
        logger.warning("Sin columna 'learning_rate' en el historial; se omite %s", ruta.name)
        return
    datos = historial.dropna(subset=["learning_rate"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(datos["step"], datos["learning_rate"], color="#55A868")
    ax.set_title("Learning rate (warmup + scheduler)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    _guardar_figura(fig, ruta)


def graficar_training_curve(historial: pd.DataFrame, ruta: Path) -> None:
    """training_curve.png: train loss y eval loss superpuestas, para ver convergencia/overfitting."""
    fig, ax = plt.subplots(figsize=(9, 5))
    hay_datos = False

    if "loss" in historial.columns:
        datos = historial.dropna(subset=["loss"])
        if not datos.empty:
            ax.plot(datos["step"], datos["loss"], label="train loss", color="#4C72B0")
            hay_datos = True

    if "eval_loss" in historial.columns:
        datos = historial.dropna(subset=["eval_loss"])
        if not datos.empty:
            ax.plot(datos["step"], datos["eval_loss"], label="eval loss", marker="s", color="#C44E52")
            hay_datos = True

    if not hay_datos:
        logger.warning("Sin datos de loss/eval_loss; se omite %s", ruta.name)
        plt.close(fig)
        return

    ax.set_title("Curva de entrenamiento: train vs. validación")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    _guardar_figura(fig, ruta)


def graficar_validation_curve(historial: pd.DataFrame, ruta: Path) -> None:
    """validation_curve.png: eval_loss (y accuracy del TripletEvaluator, si está disponible) por step."""
    if "eval_loss" not in historial.columns:
        logger.warning("Sin columna 'eval_loss' en el historial; se omite %s", ruta.name)
        return
    datos = historial.dropna(subset=["eval_loss"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(datos["step"], datos["eval_loss"], marker="s", color="#C44E52", label="eval loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Eval loss", color="#C44E52")

    # Columna de accuracy del TripletEvaluator (nombre exacto depende de la versión de la librería;
    # se busca de forma defensiva en vez de asumir la clave exacta).
    columnas_accuracy = [c for c in historial.columns if "accuracy" in c.lower()]
    if columnas_accuracy:
        columna = columnas_accuracy[0]
        datos_accuracy = historial.dropna(subset=[columna])
        if not datos_accuracy.empty:
            eje_secundario = ax.twinx()
            eje_secundario.plot(datos_accuracy["step"], datos_accuracy[columna], color="#55A868", label=columna)
            eje_secundario.set_ylabel(columna, color="#55A868")

    ax.set_title("Curva de validación")
    _guardar_figura(fig, ruta)


def generar_graficos(historial: pd.DataFrame, dir_graficos: Path) -> None:
    """Genera los 4 gráficos pedidos dentro de logs/graficos/."""
    dir_graficos.mkdir(parents=True, exist_ok=True)
    graficar_loss(historial, dir_graficos / "loss.png")
    graficar_learning_rate(historial, dir_graficos / "learning_rate.png")
    graficar_training_curve(historial, dir_graficos / "training_curve.png")
    graficar_validation_curve(historial, dir_graficos / "validation_curve.png")


def _config_a_dict() -> Dict[str, Any]:
    """Recolecta las constantes públicas (MAYÚSCULAS) de config.py, convirtiendo Path a str para JSON."""
    resultado: Dict[str, Any] = {}
    for nombre in dir(config):
        if nombre.isupper():
            valor = getattr(config, nombre)
            resultado[nombre] = str(valor) if isinstance(valor, Path) else valor
    return resultado


def guardar_config_entrenamiento(info_hardware: Dict[str, Any], ruta: Path) -> None:
    """training_config.json: fecha, hardware (GPU/CUDA/PyTorch) e hiperparámetros completos."""
    from datetime import datetime

    contenido = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "hardware": info_hardware,
        "hiperparametros": _config_a_dict(),
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2, default=str)
    logger.info("Configuración de entrenamiento guardada en: %s", ruta)


# ===============================
# EVALUACIÓN: modelo original vs. fine-tuned
# ===============================
def _ndcg_en_k(relevancia_ordenada: np.ndarray, k: int, n_relevantes: int) -> float:
    """nDCG@k con relevancia binaria: DCG@k normalizado contra el DCG del orden ideal (IDCG@k)."""
    posiciones = np.arange(1, k + 1)
    dcg = float(np.sum(relevancia_ordenada[:k] / np.log2(posiciones + 1)))
    if n_relevantes == 0:
        return 0.0
    idcg = float(np.sum(1.0 / np.log2(np.arange(1, min(k, n_relevantes) + 1) + 1)))
    return dcg / idcg if idcg > 0 else 0.0


def _construir_corpus_y_relevancia(
    df_val: pd.DataFrame,
) -> Tuple[List[str], List[str], List[set]]:
    """
    A partir de las tripletas de validación, arma: la lista de anclas únicas
    (consultas), un corpus único de documentos (positivos + negativos de todo
    el split) y, por cada ancla, el conjunto de posiciones del corpus que son
    relevantes para ella (sus propios positivos).
    """
    documentos = pd.concat([df_val["positive"], df_val["negative"]]).drop_duplicates().reset_index(drop=True)
    posicion_doc = {texto: i for i, texto in enumerate(documentos)}

    grupos = df_val.groupby("anchor")["positive"].apply(lambda serie: set(serie.unique()))
    anclas_unicas = grupos.index.tolist()
    relevantes_por_ancla = [
        {posicion_doc[texto] for texto in positivos if texto in posicion_doc} for positivos in grupos
    ]
    return anclas_unicas, documentos.tolist(), relevantes_por_ancla


def evaluar_recuperacion(
    modelo: SentenceTransformer, df_val: pd.DataFrame, k_values: List[int], batch_size: int
) -> Dict[str, float]:
    """
    Evaluación tipo "retrieval": cada ancla es una consulta contra el corpus
    de positivos+negativos de validación. Calcula Recall@k, Precision@k,
    nDCG@k, MRR y la similitud coseno promedio ancla-positivo.
    """
    anclas, documentos, relevantes_por_ancla = _construir_corpus_y_relevancia(df_val)

    emb_anclas = modelo.encode(
        anclas, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    emb_docs = modelo.encode(
        documentos, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    similitudes = emb_anclas @ emb_docs.T  # embeddings normalizados -> producto punto = similitud coseno

    acumulado: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    acumulado.update({f"precision@{k}": [] for k in k_values})
    acumulado.update({f"ndcg@{k}": [] for k in k_values})
    reciprocal_ranks: List[float] = []
    similitudes_positivo: List[float] = []

    for fila, relevantes in zip(similitudes, relevantes_por_ancla):
        orden = np.argsort(-fila)
        relevancia_ordenada = np.array([1 if pos in relevantes else 0 for pos in orden])
        n_relevantes = len(relevantes)

        posiciones_relevantes = np.flatnonzero(relevancia_ordenada)
        reciprocal_ranks.append(1.0 / (posiciones_relevantes[0] + 1) if posiciones_relevantes.size else 0.0)
        similitudes_positivo.append(float(fila[list(relevantes)].mean()) if relevantes else 0.0)

        for k in k_values:
            recuperados_relevantes = int(relevancia_ordenada[:k].sum())
            acumulado[f"recall@{k}"].append(recuperados_relevantes / n_relevantes if n_relevantes else 0.0)
            acumulado[f"precision@{k}"].append(recuperados_relevantes / k)
            acumulado[f"ndcg@{k}"].append(_ndcg_en_k(relevancia_ordenada, k, n_relevantes))

    resultado = {nombre: float(np.mean(valores)) for nombre, valores in acumulado.items()}
    resultado["mrr"] = float(np.mean(reciprocal_ranks))
    resultado["cosine_similarity_promedio"] = float(np.mean(similitudes_positivo))
    return resultado


EXPLICACIONES_METRICAS: Dict[str, str] = {
    "recall@1": "Recall@1: fracción de positivos relevantes que aparecen en el 1er resultado.",
    "recall@5": "Recall@5: fracción de positivos relevantes recuperados entre los primeros 5 resultados.",
    "recall@10": "Recall@10: ídem, entre los primeros 10.",
    "precision@1": "Precision@1: de 1 resultado devuelto, qué proporción es relevante.",
    "precision@5": "Precision@5: de los 5 resultados devueltos, qué proporción es relevante.",
    "precision@10": "Precision@10: de los 10 resultados devueltos, qué proporción es relevante.",
    "mrr": "MRR (Mean Reciprocal Rank): promedio de 1/posición del primer resultado relevante.",
    "ndcg@1": "nDCG@1: calidad del ranking en la 1ª posición, normalizada contra el orden ideal.",
    "ndcg@5": "nDCG@5: calidad del ranking entre los primeros 5 (pondera más que lo relevante esté arriba).",
    "ndcg@10": "nDCG@10: ídem, entre los primeros 10.",
    "cosine_similarity_promedio": "Similitud coseno promedio entre cada ancla y su(s) positivo(s) real(es).",
}


def comparar_modelos(df_val: pd.DataFrame, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Evalúa el modelo ORIGINAL (sin fine-tuning) y el FINE-TUNED sobre el mismo conjunto de validación."""
    logger.info("Evaluando modelo ORIGINAL (%s, sin fine-tuning)...", config.MODEL_NAME)
    modelo_original = SentenceTransformer(config.MODEL_NAME, device=str(device))
    modelo_original.max_seq_length = config.MAX_SEQ_LENGTH
    metricas_original = evaluar_recuperacion(modelo_original, df_val, config.K_VALUES, config.EVAL_ENCODE_BATCH_SIZE)
    del modelo_original
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Evaluando modelo FINE-TUNED (%s)...", config.MODEL_FINAL_DIR)
    modelo_finetuned = SentenceTransformer(str(config.MODEL_FINAL_DIR), device=str(device))
    metricas_finetuned = evaluar_recuperacion(modelo_finetuned, df_val, config.K_VALUES, config.EVAL_ENCODE_BATCH_SIZE)

    return {"original": metricas_original, "fine_tuned": metricas_finetuned}


def guardar_comparacion(comparacion: Dict[str, Dict[str, float]], ruta: Path) -> None:
    """Guarda la comparación completa (todas las métricas, ambos modelos) en JSON."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(comparacion, archivo, ensure_ascii=False, indent=2)
    logger.info("Comparación de modelos guardada en: %s", ruta)


def imprimir_comparacion(comparacion: Dict[str, Dict[str, float]]) -> None:
    """Imprime una tabla original-vs-fine-tuned con la explicación de cada métrica."""
    original, finetuned = comparacion["original"], comparacion["fine_tuned"]

    print("\n" + "=" * 78)
    print("COMPARACIÓN: MODELO ORIGINAL vs. MODELO FINE-TUNED")
    print("=" * 78)
    print(f"{'Métrica':<28}{'Original':>14}{'Fine-Tuned':>14}{'Cambio':>14}")
    for nombre in original:
        delta = finetuned[nombre] - original[nombre]
        print(f"{nombre:<28}{original[nombre]:>14.4f}{finetuned[nombre]:>14.4f}{delta:>+14.4f}")

    print("\n--- Qué mide cada métrica ---")
    for nombre, explicacion in EXPLICACIONES_METRICAS.items():
        if nombre in original:
            print(f"- {explicacion}")
    print("=" * 78)


# ===============================
# ORQUESTACIÓN
# ===============================
def main() -> None:
    fijar_semilla(config.SEED)
    device = detectar_dispositivo()
    info_hardware = mostrar_info_hardware(device)

    df = cargar_dataset_entrenamiento(config.TRAIN_FILE)
    df_train, df_val = dividir_train_validation(df, config.VALIDATION_RATIO, config.SEED)
    df_train = limitar_muestras(df_train, config.PILOT_MAX_TRAIN_EXAMPLES, config.SEED)
    df_val = limitar_muestras(df_val, config.PILOT_MAX_VAL_EXAMPLES, config.SEED)

    dataset_train, dataset_val = construir_datasets_hf(df_train, df_val)

    modelo = construir_modelo(config.MODEL_NAME, device, config.MAX_SEQ_LENGTH)
    loss = construir_loss(modelo)
    evaluador = construir_evaluador(df_val)
    args = construir_training_arguments(device)

    guardar_config_entrenamiento(info_hardware, config.RUTA_TRAINING_CONFIG_JSON)

    trainer = entrenar(modelo, args, dataset_train, dataset_val, loss, evaluador, device)

    guardar_modelo_final(trainer, config.MODEL_FINAL_DIR)
    guardar_historial(trainer, config.RUTA_TRAINING_HISTORY_JSON, config.RUTA_METRICS_CSV)

    historial = extraer_historial(trainer)
    generar_graficos(historial, config.GRAFICOS_DIR)

    comparacion = comparar_modelos(df_val, device)
    guardar_comparacion(comparacion, config.RUTA_EVALUACION_JSON)
    imprimir_comparacion(comparacion)

    print("\n" + "=" * 60)
    print("PILOTO DE FINE-TUNING COMPLETADO.")
    print(f"Modelo: SentenceTransformer('{config.MODEL_FINAL_DIR}')")
    print(f"TensorBoard: tensorboard --logdir {config.RUNS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
