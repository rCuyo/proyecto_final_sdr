# Documentación Técnica

Sistema de Recuperación Semántica de Películas — Fine-Tuning de `BAAI/bge-m3` +
ChromaDB + FastAPI

---

## 1. Arquitectura del sistema

El proyecto tiene dos partes claramente separadas en el tiempo:

**A. Pipeline offline (se corre una vez, fuera de línea)**
Descarga y limpia el dataset, construye pares/tripletas de entrenamiento, valida y
enriquece esos datos (Hard Negative Mining), y hace fine-tuning de un modelo de
embeddings (`BAAI/bge-m3`) especializado en el dominio de películas de este dataset.

**B. Sistema online (corre en cada demo/uso real)**
Con el modelo ya entrenado, se generan embeddings de las 44 471 películas y se indexan
en ChromaDB. FastAPI expone un endpoint que recibe una consulta en texto libre, la
convierte a embedding con el mismo modelo fine-tuned, y busca por similitud coseno en
ChromaDB. Un frontend HTML/JS simple consume ese endpoint.

```
┌─────────────────────────── PIPELINE OFFLINE (una vez) ───────────────────────────┐
│                                                                                    │
│  Kaggle          preparar_        generar_dataset_    validar_dataset_            │
│  "Movies          dataset.py  →   entrenamiento.py  →  entrenamiento.py           │
│  Dataset"        (limpieza)      (pares/tripletas)    (validación + Hard          │
│                                                         Negative Mining)           │
│                                          │                       │                │
│                                          ▼                       ▼                │
│                              dataset_entrenamiento.csv  dataset_entrenamiento_v2.csv│
│                                                                   │                │
│                                                                   ▼                │
│                                                          finetuning.py             │
│                                                    (SentenceTransformerTrainer,    │
│                                                     CachedMultipleNegativesRanking │
│                                                     Loss sobre BAAI/bge-m3)        │
│                                                                   │                │
│                                                                   ▼                │
│                                                  models/bge-m3-finetuned/          │
└────────────────────────────────────────────────────────────────┬─────────────────┘
                                                                   │
┌──────────────────────────── SISTEMA ONLINE (cada demo) ─────────┼─────────────────┐
│                                                                  ▼                 │
│                                                          cargar.py                 │
│                                            (encode() de las 44 471 películas       │
│                                             con el modelo fine-tuned)              │
│                                                                  │                 │
│                                                                  ▼                 │
│                                              clases_sr/chroma_db (ChromaDB)        │
│                                                                  │                 │
│                              Usuario           POST /buscar      │                 │
│                              (navegador) ───────────────────►  main.py (FastAPI)   │
│                              index.html  ◄───────────────────  (encode consulta,   │
│                                            JSON resultados       query ChromaDB)   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Descripción de cada carpeta

| Carpeta | Contenido |
|---|---|
| `scripts/` | Los 5 scripts del pipeline offline (dataset → entrenamiento) más utilidades para correr y monitorear el entrenamiento en Windows. |
| `clases_sr/` | Sistema online: API FastAPI (`main.py`), script de indexado en ChromaDB (`cargar.py`), frontend (`index.html`), la base ChromaDB de producción (`chroma_db/`) y varios scripts sueltos de prueba/exploración del curso. |
| `config.py` | Configuración central de hiperparámetros del fine-tuning (raíz del proyecto, importado por `scripts/finetuning.py`). |
| `data/processed/` | Dataset limpio y combinado (`peliculas_dataset_limpio.csv`), salida de la etapa 1. |
| `data/training/` | Pares positivos/negativos, tripletas y datasets de entrenamiento (v1 y v2), salida de las etapas 2 y 3. |
| `data/reports/` | Reporte de validación del dataset (`dataset_report.txt`, `dataset_statistics.json`) y sus gráficos (`graficos/`). |
| `models/_checkpoints/` | Checkpoints intermedios del `Trainer` durante el fine-tuning (no es el modelo final). |
| `models/bge-m3-finetuned/` | Modelo final fine-tuned, listo para cargar con `SentenceTransformer(...)`. |
| `logs/` | Historial completo del entrenamiento: log de texto, métricas por step (CSV/JSON), configuración usada, comparación de métricas original vs. fine-tuned, y gráficos de convergencia (`graficos/`). |
| `reports/` | Reporte final de comparación de modelos (`comparacion_modelos.md`) y una base ChromaDB temporal usada solo para esa comparación. |

---

## 3. Descripción de cada script

### `scripts/preparar_dataset.py` — Etapa 1: dataset

Descarga (vía `kagglehub`) los 3 archivos necesarios de "The Movies Dataset"
(`movies_metadata.csv`, `keywords.csv`, `credits.csv`), reporta valores nulos/duplicados
por archivo, y produce `data/processed/peliculas_dataset_limpio.csv`: un único CSV con
44 471 películas, con columnas derivadas legibles (`categoria`, `director`, `reparto`,
`descripcion`) parseadas desde las columnas JSON-like originales del dataset de Kaggle.

### `scripts/generar_dataset_entrenamiento.py` — Etapa 2: pares y tripletas

Construye la columna `texto` (formato fijo: Título / Categoría / Director / Reparto /
Palabras clave / Descripción) que el modelo va a aprender a embeber. Sobre ese texto:

- **Pares positivos**: para cada película-ancla, busca candidatas que compartan al
  menos un género (bloqueo por índice invertido, no fuerza bruta O(n²)) y las puntúa con
  un score combinado (género 30% + keywords 25% + TF-IDF 20% + director 15% + reparto
  10%), quedándose con las 2 mejores que superen el umbral 0.35.
- **Pares negativos**: muestrea candidatas al azar y exige género disjunto, casi cero
  keywords en común, TF-IDF bajo y distinto director — negativos "fáciles" e
  inequívocos.
- **Tripletas**: empareja cada positivo con un negativo de la misma ancla, listas para
  `(anchor, positive, negative)`.

Guarda `pares_positivos.csv`, `pares_negativos.csv`, `tripletas.csv` y
`dataset_entrenamiento.csv` en `data/training/`.

### `scripts/validar_dataset_entrenamiento.py` — Etapa 3: validación + Hard Negative Mining

Reutiliza las funciones y pesos de la etapa 2 (no los reimplementa) para:

1. **Validar** el dataset (conteos, campos vacíos, duplicados, integridad de tripletas,
   cobertura) y guardar un reporte (`dataset_report.txt`, `dataset_statistics.json`).
2. **Mejorar cobertura de positivos**: para películas que quedaron sin ningún positivo
   en la etapa 2, toma el mejor candidato disponible para *esa* película puntual (no
   baja el umbral global) si supera un piso mínimo de calidad.
3. **Generar gráficos** (top géneros/keywords/directores, distribución de longitud de
   texto, tripletas por ancla) en `data/reports/graficos/`.
4. **Hard Negative Mining**: para cada par (ancla, positivo), busca un negativo *difícil*
   — con similitud TF-IDF alta pero siempre por debajo de la del positivo real menos un
   margen (0.02), garantizando que el modelo tenga ejemplos difíciles sin ejemplos
   contradictorios.
5. Combina tripletas fáciles + hard en `dataset_entrenamiento_v2.csv`, el dataset final
   que consume el fine-tuning.

### `scripts/finetuning.py` — Etapa 4: fine-tuning

Carga `dataset_entrenamiento_v2.csv`, separa train/validación por ancla única (90/10,
sin fuga de información), entrena `BAAI/bge-m3` con
`SentenceTransformerTrainer` + `CachedMultipleNegativesRankingLoss` (con GradCache para
que el batch efectivo no dependa de la VRAM disponible), guarda el mejor checkpoint como
modelo final, genera los gráficos de entrenamiento, y finalmente evalúa y compara
Recall/Precision/nDCG/MRR entre el modelo original y el fine-tuned sobre el mismo
conjunto de validación. Ver detalle completo en la sección 7.

### `clases_sr/cargar.py` — Indexado en ChromaDB

Reutiliza `cargar_dataset` + `crear_columna_texto` de la etapa 2 (mismo texto que se usó
para entrenar), borra la base ChromaDB anterior, genera embeddings de las 44 471
películas con el modelo fine-tuned, y las inserta en la colección `peliculas` de
`clases_sr/chroma_db/`. Verifica al final que documentos, embeddings e IDs coincidan en
cantidad, sin duplicados ni vacíos.

### `clases_sr/main.py` — API FastAPI

Carga el modelo fine-tuned y la colección `peliculas` de ChromaDB al arrancar. Expone:

- `GET /` — sirve el frontend (ver nota en la sección 6 de `AUDITORIA.md` sobre una
  inconsistencia de ruta).
- `POST /buscar` — recibe `{consulta, n_resultados}`, genera el embedding de la consulta
  y devuelve las películas más similares (título, categoría, descripción).

### `scripts/verificar_migracion.py` — Verificación end-to-end + reporte final

Levanta la API real (`uvicorn main:app`), le manda consultas de prueba reales, y
compara — para las mismas consultas — el modelo original vs. el fine-tuned, cada uno
sobre su propia colección ChromaDB (construye una colección temporal solo para el
modelo original, en `reports/_chroma_comparacion_original/`). Con todo eso, más las
métricas ya calculadas por `finetuning.py`, genera `reports/comparacion_modelos.md`.

### Scripts de conveniencia (Windows)

- `scripts/ejecutar_finetuning.bat` / `.ps1`: activan el entorno virtual si existe y
  corren `scripts/finetuning.py`; si detectan Windows Terminal, abren automáticamente 3
  pestañas (entrenamiento, TensorBoard, monitoreo de GPU).
- `scripts/abrir_tensorboard.bat`: levanta TensorBoard apuntando a la carpeta `runs/`.
- `scripts/monitorear_gpu.ps1`: muestra uso de GPU/VRAM/temperatura en vivo vía
  `nvidia-smi`, refrescando cada 2 segundos.

---

## 4. Flujo completo del proyecto

```
1. preparar_dataset.py                  → data/processed/peliculas_dataset_limpio.csv
2. generar_dataset_entrenamiento.py     → data/training/{pares_*, tripletas, dataset_entrenamiento}.csv
3. validar_dataset_entrenamiento.py     → data/training/dataset_entrenamiento_v2.csv + data/reports/*
4. finetuning.py (usa config.py)        → models/bge-m3-finetuned/ + logs/* + reports parciales
5. clases_sr/cargar.py                  → clases_sr/chroma_db/ (colección "peliculas")
6. clases_sr/main.py (uvicorn)          → API en http://127.0.0.1:8000
7. clases_sr/index.html                 → frontend consumiendo la API
8. scripts/verificar_migracion.py       → reports/comparacion_modelos.md (verificación + reporte final)
```

Cada etapa deja artefactos en disco que la siguiente etapa consume; no hay estado en
memoria compartido entre scripts, así que cada paso se puede re-ejecutar de forma
independiente si sus insumos ya existen.

---

## 5. Tecnologías utilizadas

| Categoría | Tecnología |
|---|---|
| Modelo de embeddings | `BAAI/bge-m3` (SentenceTransformers) |
| Fine-tuning | `sentence-transformers` (`SentenceTransformerTrainer`), `transformers`, `accelerate`, `datasets`, PyTorch (CUDA) |
| Pérdida de entrenamiento | `CachedMultipleNegativesRankingLoss` (GradCache) |
| Preparación de datos | `pandas`, `numpy`, `scikit-learn` (TF-IDF, similitud coseno) |
| Base vectorial | ChromaDB (`PersistentClient`) |
| Backend / API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript vanilla (sin framework) |
| Monitoreo de entrenamiento | TensorBoard, `nvidia-smi` |
| Visualizaciones | Matplotlib |
| Origen del dataset | Kaggle — "The Movies Dataset" (`kagglehub`) |

---

## 6. Modelo utilizado

**Base:** `BAAI/bge-m3`, un modelo de embeddings multilingüe de 568M de parámetros,
capaz de generar vectores de 1024 dimensiones y de manejar secuencias largas (hasta
8192 tokens, acotado a 256 en este proyecto porque cubre el percentil 95 de longitud de
los textos del dataset).

**Fine-tuned:** el mismo modelo, ajustado sobre el dominio específico de este dataset de
películas (44 471 títulos), guardado en `models/bge-m3-finetuned/`, cargable
directamente con:

```python
from sentence_transformers import SentenceTransformer
modelo = SentenceTransformer("models/bge-m3-finetuned")
```

---

## 7. Proceso de Fine-Tuning

**Formato de entrada:** tripletas `(anchor, positive, negative)`, cada elemento con el
texto estructurado "Título / Categoría / Director / Reparto / Palabras clave /
Descripción".

**Función de pérdida:** `CachedMultipleNegativesRankingLoss` — el mismo objetivo
contrastivo con el que se entrenó BGE originalmente, con GradCache para desacoplar el
batch efectivo (`BATCH_SIZE=32`, calidad del entrenamiento) del pico de VRAM
(`MINI_BATCH_SIZE=4`, lo que realmente se procesa a la vez en la GPU).

**Este es un PILOTO**, no el entrenamiento definitivo — así lo documenta explícitamente
`config.py`. Corrió con un subconjunto del dataset para verificar rápidamente que el
pipeline converge y mejora sobre el modelo original antes de invertir en un
entrenamiento completo.

**Hiperparámetros usados** (`config.py`, corrida real registrada en
`logs/training_config.json`):

| Parámetro | Valor |
|---|---|
| Épocas | 1 |
| Ejemplos de entrenamiento (piloto) | 2 000 |
| Ejemplos de validación (piloto) | 400 |
| Batch efectivo / mini-batch | 32 / 4 |
| Learning rate | 2e-5 (warmup 10%) |
| Optimizador | Adafactor (menor huella de memoria que AdamW) |
| Precisión | FP16 |
| Gradient checkpointing | Sí |
| `max_seq_length` | 256 tokens |
| Early stopping | 3 evaluaciones sin mejora |
| Semilla | 42 |

**Hardware real de la corrida:** NVIDIA GeForce RTX 4060 Ti (8 GB VRAM), PyTorch
2.11.0+cu128, CUDA 12.8. La detección de GPU es siempre automática en tiempo de
ejecución (`finetuning.detectar_dispositivo`).

**Split train/validación:** 90/10 por **ancla única** (no por fila), para que todas las
tripletas de una misma película caigan del mismo lado y no haya fuga de información
entre entrenamiento y validación.

**Salidas del entrenamiento** (todas en `logs/` y `models/`):
`models/bge-m3-finetuned/` (modelo final), `logs/training.log`,
`logs/training_history.json` / `metrics.csv` (historial completo de loss/lr/VRAM por
step), `logs/graficos/{loss,learning_rate,training_curve,validation_curve}.png`,
`logs/evaluacion_comparativa.json` (métricas finales original vs. fine-tuned).

---

## 8. Proceso de construcción de ChromaDB

`clases_sr/cargar.py` reutiliza exactamente la misma función de carga y de construcción
de texto (`cargar_dataset` + `crear_columna_texto`, de la etapa 2) que se usó para
generar los datos de entrenamiento — así el texto que se embebe para búsqueda es
idéntico en formato al texto con el que se hizo el fine-tuning. Pasos:

1. Elimina la base ChromaDB anterior por completo (no se reutiliza ningún embedding
   viejo).
2. Crea un cliente `PersistentClient` nuevo y la colección `peliculas`.
3. Genera embeddings de las 44 471 películas con el modelo fine-tuned, normalizados
   (para que la distancia L2 de ChromaDB equivalga a similitud coseno), en lotes de 64.
4. Inserta documentos + embeddings + metadata (título, categoría, sinopsis original) en
   lotes de 1000 (límite práctico de `add()` en ChromaDB).
5. Verifica que documentos, embeddings, IDs y dimensión (1024) coincidan, sin
   duplicados ni vacíos.

Resultado verificado: **44 471 documentos indexados**, dimensión 1024, sin
inconsistencias.

---

## 9. Funcionamiento de la API

FastAPI (`clases_sr/main.py`), con CORS abierto (`allow_origins=["*"]`) para que el
frontend estático pueda consumirla sin restricciones durante la demo.

**Al arrancar:** carga el modelo fine-tuned una sola vez (`SentenceTransformer(...)`),
abre la colección `peliculas` de ChromaDB, construye el `RemoteEmbeddingProvider` (modo
online, sin validar el token todavía) y abre la colección `peliculas_original` — todo
queda en memoria, no se recarga por request. Ver detalle completo del modo Online/Local
en la sección 14.

**`POST /buscar`**

Entrada:
```json
{ "consulta": "películas sobre viajes en el tiempo", "n_resultados": 5, "modo": "local" }
```
`modo` es opcional (`"local"` por defecto, valores válidos `"local"` / `"online"`) —
preserva el comportamiento de antes de la sección 14 para cualquier cliente que no lo
envíe.

Proceso: genera el embedding de la consulta con el proveedor correspondiente a `modo`
(`normalize_embeddings=True` en modo local; ver sección 14), consulta la colección
ChromaDB que le corresponde a ese modo por los `n_resultados` vecinos más cercanos, y
devuelve título, categoría y descripción de cada resultado.

Salida (idéntica sin importar el modo usado — `modo` no aparece en la respuesta):
```json
{ "consulta": "...", "cantidad": 5, "resultados": [ {"titulo": "...", "categoria": "...", "descripcion": "..."} ] }
```

Errores del pipeline de inferencia (de cualquiera de los dos modos) se capturan y
devuelven como `HTTP 500` con el detalle del error; un `modo` inválido es rechazado
antes, por validación de Pydantic, como `HTTP 422`.

**Cómo levantarla:**
```bash
cd clases_sr
uvicorn main:app --reload
```
Queda disponible en `http://127.0.0.1:8000` (documentación automática interactiva en
`http://127.0.0.1:8000/docs`).

---

## 10. Funcionamiento del Frontend

`clases_sr/index.html`: página única (HTML + CSS + JS vanilla, sin build step ni
framework), con estética de "boletería de cine" (marquesina, ticket, tira de
fotogramas). Flujo:

1. El usuario escribe una consulta en lenguaje natural, elige cuántos resultados quiere
   (3/5/10/20) y elige el modo (Local/Online, selector `#modo` — ver sección 14).
2. Al enviar el formulario, hace `fetch` a `POST http://127.0.0.1:8000/buscar` con la
   consulta, `n_resultados` y `modo`.
3. Mientras espera, muestra un estado de carga; si la API no responde, distingue el
   caso "no se pudo conectar" (CORS/servidor caído) del resto de errores.
4. Renderiza cada resultado como una tarjeta (índice, título, categoría), y muestra la
   descripción completa en un tooltip que sigue al cursor al pasar por encima de la
   tarjeta.

La URL de la API está hardcodeada en el JS (`API_BASE_URL = "http://127.0.0.1:8000"`,
línea 326) — si la API corre en otro host/puerto, hay que editar esa constante.

**Cómo usarlo:** con la API corriendo, abrir `clases_sr/index.html` directamente en el
navegador (no requiere servidor propio).

---

## 11. Configuración del proyecto

Toda la configuración del fine-tuning vive centralizada en `config.py` (raíz del
proyecto), documentada e importada únicamente por `scripts/finetuning.py`. Ningún otro
script de las etapas 1-3 ni el sistema online la importa. Los bloques son:

- **Rutas**: modelo base, dataset de entrenamiento, directorios de salida (modelos,
  logs, gráficos, runs de TensorBoard).
- **Datos/split**: semilla, proporción de validación, límites del piloto
  (`PILOT_MAX_TRAIN_EXAMPLES` / `PILOT_MAX_VAL_EXAMPLES` — poner en `None` para usar el
  dataset completo).
- **Modelo**: `MAX_SEQ_LENGTH`.
- **Entrenamiento**: épocas, batch sizes, learning rate, optimizador, precisión
  (FP16/BF16), gradient checkpointing.
- **Checkpoints/logging/early stopping**.
- **Evaluación**: valores de k para Recall/Precision/nDCG.
- **Hardware**: la detección de CUDA es siempre automática; `FORZAR_CPU` solo documenta
  la intención de nunca asumir CPU por defecto.

No hay archivo `.env` en el proyecto; la única credencial externa que aparece es el
token de Hugging Face hardcodeado en dos scripts de prueba de `clases_sr/` — **no** en
el sistema en producción (`main.py` usa el modelo local). Ver `AUDITORIA.md` sección 0
para la recomendación al respecto antes de compartir el repositorio.

---

## 12. Cómo ejecutar el proyecto desde cero

### Requisitos previos

- Python 3.13 (el proyecto se desarrolló y verificó con 3.13.5).
- GPU NVIDIA con drivers CUDA si se quiere re-entrenar (no es necesario solo para
  servir la API con el modelo ya entrenado).
- Credenciales de Kaggle (`~/.kaggle/kaggle.json` o `KAGGLE_USERNAME` /
  `KAGGLE_KEY`) — solo si se va a volver a descargar el dataset desde cero.

### Entorno

Este proyecto usa **dos entornos Python** en la práctica (ver `AUDITORIA.md` sección 6):

- Un entorno para el pipeline de datos y el entrenamiento (`torch`, `sentence-transformers`,
  `transformers`, `accelerate`, `datasets`, `scikit-learn`, `pandas`, `numpy`,
  `matplotlib`, `tensorboard`, `kagglehub`) — en este proyecto es `.venv/`.
- Un entorno (o el mismo, si se prefiere unificar) con `fastapi`, `uvicorn`, `chromadb`,
  `requests` para servir la API.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r scripts\requirements.txt
# torch con soporte CUDA se instala aparte, según la GPU disponible:
# https://pytorch.org/get-started/locally/
```

### Pasos, en orden

```bash
# 1. Dataset
python scripts\preparar_dataset.py

# 2. Pares y tripletas
python scripts\generar_dataset_entrenamiento.py

# 3. Validación + Hard Negative Mining
python scripts\validar_dataset_entrenamiento.py

# 4. Fine-tuning (o usar scripts\ejecutar_finetuning.bat / .ps1 en Windows)
python scripts\finetuning.py

# 5. Construir la base vectorial con el modelo fine-tuned
python clases_sr\cargar.py

# 6. Levantar la API
cd clases_sr
uvicorn main:app --reload

# 7. Abrir clases_sr\index.html en el navegador

# 8. (Opcional) Verificación end-to-end + reporte de comparación
python scripts\verificar_migracion.py
```

Cada script valida sus propios insumos y lanza un error claro indicando qué script
anterior falta correr si algo no está listo.

### Monitoreo durante el entrenamiento (opcional, Windows)

```bash
scripts\ejecutar_finetuning.bat
```
Si Windows Terminal está instalado, abre automáticamente 3 pestañas: entrenamiento,
TensorBoard y monitoreo de GPU. Ver la nota sobre la ruta de `runs/` en `AUDITORIA.md`
sección 6 si TensorBoard no muestra la corrida esperada.

---

## 13. Cómo volver a entrenar el modelo en el futuro

El pipeline ya está pensado para esto — este entrenamiento fue explícitamente un
**piloto**. Para pasar al entrenamiento definitivo:

1. En `config.py`, poner en `None` los límites del piloto:
   ```python
   PILOT_MAX_TRAIN_EXAMPLES: Optional[int] = None
   PILOT_MAX_VAL_EXAMPLES: Optional[int] = None
   ```
   Esto usa el dataset completo (23 857 tripletas fáciles + 3 643 hard, no solo el
   subconjunto de 2000/400 del piloto).

2. Subir `EPOCHS` (1 fue suficiente para verificar convergencia, no para un
   entrenamiento definitivo — evaluar con la curva de `logs/graficos/validation_curve.png`
   cuántas épocas hacen falta antes de que `eval_loss` deje de mejorar; `EarlyStoppingCallback`
   ya está configurado para frenar automáticamente).

3. Si se dispone de más VRAM que los 8 GB de la RTX 4060 Ti original, se puede:
   - Cambiar `OPTIMIZER_NAME` de `"adafactor"` a `"adamw_torch"`.
   - Subir `MINI_BATCH_SIZE` (el que realmente determina el pico de VRAM, no
     `BATCH_SIZE`).

4. Si se quiere reanudar una corrida interrumpida en vez de empezar de cero,
   `finetuning.py` ya detecta automáticamente el checkpoint de mayor step en
   `models/_checkpoints/` y continúa desde ahí (`detectar_checkpoint`) — no requiere
   ninguna acción manual.

5. Volver a correr `python scripts\finetuning.py` (o el `.bat`/`.ps1`). Al terminar,
   sobrescribe `models/bge-m3-finetuned/` con el nuevo mejor modelo.

6. Reconstruir la base vectorial con el nuevo modelo: `python clases_sr\cargar.py`
   (esto borra y recrea `clases_sr/chroma_db/` por completo).

7. (Opcional pero recomendado) Volver a correr `python scripts\verificar_migracion.py`
   para regenerar `reports/comparacion_modelos.md` con las métricas del nuevo modelo.

Si en el futuro se agregan más películas al dataset original, hay que volver a correr
el pipeline completo desde la etapa 1 (`preparar_dataset.py`), porque los pares
positivos/negativos y el Hard Negative Mining dependen del corpus completo (TF-IDF e
índices invertidos se recalculan sobre todo el dataset).

---

## 14. Modo Online vs. Modo Local

Etapa final del proyecto: la misma aplicación (una sola API, un solo FastAPI, un solo
frontend, un solo endpoint `/buscar`) puede generar el embedding de la consulta de dos
formas intercambiables, elegidas por el usuario en tiempo de request.

### 14.1 Los dos flujos

```
MODO LOCAL (por defecto)
Usuario → index.html → POST /buscar {modo:"local"}  → LocalEmbeddingProvider
  → SentenceTransformer(models/bge-m3-finetuned).encode()
  → ChromaDB: colección "peliculas" (clases_sr/chroma_db)
  → Resultados

MODO ONLINE
Usuario → index.html → POST /buscar {modo:"online"} → RemoteEmbeddingProvider
  → InferenceClient(provider="hf-inference").feature_extraction(texto, model="BAAI/bge-m3")
  → Hugging Face Inference API (modelo BAAI/bge-m3 original, sin fine-tuning)
  → ChromaDB: colección "peliculas_original" (reports/_chroma_comparacion_original)
  → Resultados
```

### 14.2 Qué código del docente se reutilizó

El patrón exacto de `clases_sr/consulta1.py` / `clases_sr/pruebatoken.py` —
`InferenceClient(provider="hf-inference", api_key=...)` seguido de
`client.feature_extraction(texto, model="BAAI/bge-m3")` — se portó tal cual a
`RemoteEmbeddingProvider.generar_embedding()` en el nuevo módulo
`clases_sr/embedding_providers.py`. No se reescribió la llamada al modelo remoto: se
encapsuló en una clase para poder intercambiarla con el modo local sin `if/else`
dispersos.

Único cambio consciente respecto al código original del docente: el token ya no queda
hardcodeado en el archivo (`consulta1.py`/`pruebatoken.py` lo tenían así, y ese hallazgo
ya está señalado como riesgo de seguridad en `AUDITORIA.md`). `RemoteEmbeddingProvider`
recibe el token por parámetro; `main.py` se lo pasa leyéndolo de la variable de entorno
`HF_TOKEN` (`os.environ.get("HF_TOKEN")`). Si no está configurada, la app arranca igual
—el modo local sigue funcionando sin requisito nuevo— y el error solo aparece si alguien
efectivamente elige modo online sin token, como `HTTP 500` (mismo manejo de excepciones
que ya existía).

### 14.3 Qué código propio se reutilizó

- El flujo local existente en `main.py` (`SentenceTransformer(RUTA_MODELO).encode(...,
  normalize_embeddings=True)`) se envolvió, sin reescribirlo, en `LocalEmbeddingProvider`.
- La colección `peliculas_original` **no se generó de nuevo**: es el mismo artefacto que
  `scripts/verificar_migracion.py` ya había construido (función
  `construir_coleccion_original()`) para comparar el modelo original contra el
  fine-tuned. Se reutiliza el archivo ya generado en
  `reports/_chroma_comparacion_original/`, cero cómputo adicional, cero llamadas nuevas
  a Hugging Face para reconstruir nada.
- Todo el resto de `main.py` (CORS, `ConsultaRequest`, armado de la respuesta,
  `HTTPException`) permanece literalmente igual a como estaba antes de esta etapa.

### 14.4 Por qué dos colecciones de ChromaDB, no una

`clases_sr/chroma_db` (colección `peliculas`) fue embebida por `cargar.py` con **nuestro
modelo fine-tuned**. Si el modo online generara el embedding de la consulta con el
modelo original y lo comparara contra esa misma colección, estaría comparando dos
espacios de embeddings distintos — ChromaDB nunca "falla" (siempre devuelve algo), pero
el resultado no es una búsqueda por similitud válida.

Esto ya lo había resuelto el propio proyecto una vez: el docstring de
`construir_coleccion_original()` en `scripts/verificar_migracion.py` dice
textualmente *"cada modelo con SU PROPIO espacio de embeddings, no mezclando uno con
las distancias del otro"*. Por eso el modo online usa `peliculas_original`
(embebida con el modelo original) en vez de `peliculas`.

`main.py` abre dos `chromadb.PersistentClient` (uno por carpeta) — sigue siendo una
sola aplicación, un solo FastAPI, un solo endpoint; es un detalle interno de qué
colección consultar según el modo, invisible para el frontend y para el contrato de la
API.

Nota de metadata: `peliculas_original` fue construida solo con `{titulo, categoria}`
(sin `descripcion`, porque `verificar_migracion.py` no la necesitaba para su propio
propósito de comparación). El armado de la respuesta en `main.py` usa
`meta.get("descripcion", "")` en vez de `meta["descripcion"]` para que ambas
colecciones sean compatibles con el mismo código, sin modificar los datos ya
guardados.

### 14.5 Capa de abstracción (`clases_sr/embedding_providers.py`)

```python
class EmbeddingProvider(ABC):
    def generar_embedding(self, texto: str) -> list[float]: ...

class LocalEmbeddingProvider(EmbeddingProvider): ...   # envuelve el modelo ya cargado
class RemoteEmbeddingProvider(EmbeddingProvider): ...  # envuelve InferenceClient
```

`main.py` arma, una sola vez al iniciar, dos diccionarios indexados por `modo`:

```python
PROVEEDORES = {"local": LocalEmbeddingProvider(modelo), "online": RemoteEmbeddingProvider(api_key=...)}
COLECCIONES  = {"local": collection, "online": collection_original}
```

Y el endpoint `/buscar` selecciona ambos con una sola línea cada uno
(`PROVEEDORES[data.modo]`, `COLECCIONES[data.modo]`) — es el único punto de decisión
entre los dos modos en todo el proyecto. Agregar un tercer proveedor en el futuro no
requeriría tocar el endpoint, solo una clase nueva y una entrada en cada diccionario.

`ConsultaRequest` incorpora `modo: Literal["local", "online"] = "local"` — el default
preserva el comportamiento previo a esta etapa para cualquier cliente que no envíe el
campo.

### 14.6 Selector de modo en el frontend

`clases_sr/index.html` agrega un `<select id="modo" class="n-select">` (Local/Online,
"Local" seleccionado por defecto) dentro del mismo `.meta-row` donde ya vivía el
selector de cantidad de resultados — reutiliza la clase CSS existente, sin estilos
nuevos. El JS de `buscarPeliculas(...)` agrega `modo` al cuerpo del `fetch`; la
respuesta del backend no cambia (`modo` es un campo de entrada, no de salida), así que
el resto del renderizado de resultados no se modificó.

### 14.7 Archivos modificados vs. no modificados en esta etapa

**Modificados:**
- `clases_sr/main.py` — import del nuevo módulo, segundo `PersistentClient`, diccionarios
  `PROVEEDORES`/`COLECCIONES`, campo `modo` en `ConsultaRequest`, las 3 líneas del cuerpo
  de `buscar_peliculas`, `meta.get("descripcion", "")`, y `os.environ.setdefault("USE_TF",
  "0")` antes de importar `sentence_transformers` (fix de un problema de entorno
  preexistente —TensorFlow instalado, incompatible con Keras 3— no relacionado con esta
  etapa, pero necesario para poder levantar el servidor; mismo workaround que ya usaba
  `scripts/finetuning.py`).
- `clases_sr/index.html` — selector de modo + JS.
- `scripts/requirements.txt` — se agregó `huggingface_hub` como dependencia directa
  documentada (ya estaba instalada de forma transitiva).
- `DOCUMENTACION_TECNICA.md` — esta sección.

**Creados:**
- `clases_sr/embedding_providers.py`.

**NO modificados:**
- `clases_sr/cargar.py` — sigue construyendo únicamente `peliculas` con el modelo
  fine-tuned, exactamente igual que antes de esta etapa.
- `config.py` — fuera de alcance (exclusivo del fine-tuning; `main.py` nunca lo importó).
- `clases_sr/consulta1.py`, `consulta2.py`, `pruebatoken.py`, `ubica_modelo.py`,
  `prueba_fastapi.py` — quedan intactos como referencia; su lógica se portó a
  `embedding_providers.py`, no se editó ni se eliminó ahí.
- `scripts/verificar_migracion.py` y el resto del pipeline de datos/entrenamiento
  (`preparar_dataset.py`, `generar_dataset_entrenamiento.py`,
  `validar_dataset_entrenamiento.py`, `finetuning.py`) — sin relación con este cambio.
- `reports/_chroma_comparacion_original/` — se lee, no se regenera ni se modifica.

### 14.8 Verificación realizada

Se probaron las 6 consultas de referencia del proyecto (`robots`, `piratas`, `terror
psicológico`, `astronautas`, `amor imposible`, `viajes en el tiempo`) contra `POST
/buscar` en ambos modos, con resultado `HTTP 200` y forma de respuesta idéntica en los
12 casos. También se verificó: `modo` omitido se comporta como `"local"` (compatibilidad
hacia atrás), un `modo` inválido responde `HTTP 422` sin afectar al servidor, y el modo
online sin `HF_TOKEN` configurado responde `HTTP 500` con el detalle del error sin
romper el modo local en la misma corrida del servidor.

### 14.9 Cómo usar el modo online

Requiere una variable de entorno `HF_TOKEN` con un token válido de Hugging Face antes de
levantar la API:

```bash
set HF_TOKEN=tu_token_aqui          REM cmd
$env:HF_TOKEN = "tu_token_aqui"     # PowerShell

cd clases_sr
uvicorn main:app --reload
```

Sin `HF_TOKEN` configurado, el modo local funciona exactamente igual que siempre; el
modo online devuelve `HTTP 500` recién si el usuario lo selecciona.
