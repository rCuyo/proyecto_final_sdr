"""
Configuración central del piloto de Fine-Tuning de BAAI/bge-m3.

Toda la configuración del entrenamiento vive acá (etapa 4 del proyecto). Ningún
otro script de las etapas anteriores se modifica ni se importa desde acá:
este módulo es autónomo, pensado para ser leído/ajustado por `scripts/finetuning.py`.

Es un PILOTO: los defaults están pensados para correr rápido en una RTX 4060 Ti
de 8GB VRAM y verificar que el pipeline funciona y converge, no para el
entrenamiento definitivo. Los valores marcados "PILOTO" son los primeros que
hay que cambiar (o poner en None) al pasar al entrenamiento completo.
"""

from pathlib import Path
from typing import Optional

# ===============================
# RUTAS
# ===============================
BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "BAAI/bge-m3"
TRAIN_FILE = BASE_DIR / "data" / "training" / "dataset_entrenamiento_v2.csv"

MODELS_DIR = BASE_DIR / "models"
# Directorio de trabajo del Trainer (checkpoints intermedios, NO el modelo final).
OUTPUT_DIR = MODELS_DIR / "_checkpoints"
# Modelo final, limpio, cargable directo con SentenceTransformer(MODEL_FINAL_DIR).
MODEL_FINAL_DIR = MODELS_DIR / "bge-m3-finetuned"

LOGS_DIR = BASE_DIR / "logs"
GRAFICOS_DIR = LOGS_DIR / "graficos"
RUNS_DIR = BASE_DIR / "runs"  # TensorBoard: tensorboard --logdir runs

RUTA_TRAINING_LOG = LOGS_DIR / "training.log"
RUTA_METRICS_CSV = LOGS_DIR / "metrics.csv"
RUTA_TRAINING_HISTORY_JSON = LOGS_DIR / "training_history.json"
RUTA_TRAINING_CONFIG_JSON = LOGS_DIR / "training_config.json"
RUTA_EVALUACION_JSON = LOGS_DIR / "evaluacion_comparativa.json"

# ===============================
# DATOS / SPLIT
# ===============================
SEED = 42
VALIDATION_RATIO = 0.10  # fracción de anclas (no de filas) reservada para validación

# PILOTO: limita el tamaño para que la corrida sea rápida. Poner en None para
# usar el dataset completo en el entrenamiento definitivo.
PILOT_MAX_TRAIN_EXAMPLES: Optional[int] = 2000
PILOT_MAX_VAL_EXAMPLES: Optional[int] = 400

# ===============================
# MODELO
# ===============================
MAX_SEQ_LENGTH = 256  # cubre el p95 de longitud de nuestros textos (~946 caracteres ≈ 200 tokens)

# ===============================
# ENTRENAMIENTO
# ===============================
EPOCHS = 1  # PILOTO: 1 época alcanza para verificar convergencia
BATCH_SIZE = 32  # batch "efectivo" (negativos en batch) que ve la pérdida CachedMNRL
# Tamaño real de sub-lote que se procesa por vez en la GPU (GradCache): esto es
# lo que realmente determina el pico de VRAM, no BATCH_SIZE. Conservador por
# los 8GB de VRAM disponibles.
MINI_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1  # subir esto (no MINI_BATCH_SIZE) para simular batches aún mayores

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# AdamW estándar (2x params en estado del optimizador) más params+grads no entra
# holgadamente en 8GB junto con activaciones para un modelo de 568M parámetros.
# Adafactor (~1x params de estado) libera memoria para el resto del pipeline.
# Cambiar a "adamw_torch" si se dispone de más VRAM (ej. variante de 16GB).
OPTIMIZER_NAME = "adafactor"
USE_GRADIENT_CHECKPOINTING = True  # recomputa activaciones en vez de guardarlas: baja pico de VRAM

USE_FP16 = True   # RTX 4060 Ti (Ada Lovelace) soporta FP16 nativamente
USE_BF16 = False  # alternativa igualmente soportada por la GPU; dejar una sola activa

# ===============================
# CHECKPOINTS / LOGGING / EARLY STOPPING
# ===============================
SAVE_STEPS = 50   # también se usa como eval_steps (HF Trainer exige que coincidan)
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2  # cuántos checkpoints intermedios conservar en OUTPUT_DIR
EARLY_STOPPING_PATIENCE = 3  # evaluaciones sin mejora antes de frenar

# ===============================
# EVALUACIÓN (Recall/Precision/MRR/nDCG)
# ===============================
K_VALUES = [1, 5, 10]
EVAL_ENCODE_BATCH_SIZE = 32

# ===============================
# HARDWARE
# ===============================
# La detección de CUDA es SIEMPRE automática en tiempo de ejecución
# (ver `finetuning.detectar_dispositivo`); esta constante no fuerza nada,
# solo documenta la intención de nunca asumir CPU por defecto.
FORZAR_CPU = False
