# 🎬 Recomendador Semántico de Películas

Sistema de búsqueda semántica de películas por lenguaje natural, construido sobre un
modelo de embeddings (`BAAI/bge-m3`) fine-tuneado específicamente sobre un dataset de
44 471 películas, servido con ChromaDB + FastAPI y consumido por un frontend web simple.

> Documentación técnica completa: [`DOCUMENTACION_TECNICA.md`](./DOCUMENTACION_TECNICA.md)
> Resultados y métricas detalladas: [`RESULTADOS.md`](./RESULTADOS.md)
> Resumen para presentación: [`RESUMEN_PROYECTO.md`](./RESUMEN_PROYECTO.md)

---

## Descripción del proyecto

En vez de buscar películas por coincidencia exacta de palabras, el sistema entiende el
*significado* de la consulta. Preguntas como *"películas sobre viajes en el tiempo"* o
*"terror psicológico"* devuelven resultados semánticamente relevantes, aunque el título
o la sinopsis no contengan esas palabras textualmente.

El corazón del proyecto es un modelo de embeddings de propósito general
(`BAAI/bge-m3`) al que se le hizo **fine-tuning** sobre pares y tripletas construidos
automáticamente a partir del propio dataset de películas (género, keywords, director,
reparto y similitud textual), incluyendo **Hard Negative Mining** para forzar al modelo
a distinguir casos difíciles.

## Objetivo

Demostrar, de punta a punta, que un modelo de embeddings pre-entrenado genérico mejora
de forma medible en una tarea de recuperación semántica de dominio específico (búsqueda
de películas) cuando se le hace fine-tuning con datos construidos a partir del propio
corpus, y desplegar ese modelo en un sistema real y consultable (API + frontend).

## Características

- Pipeline completo y reproducible: descarga → limpieza → generación de pares/tripletas
  → validación → Hard Negative Mining → fine-tuning → indexado vectorial → API → frontend.
- Fine-tuning con `CachedMultipleNegativesRankingLoss` (GradCache), pensado para correr
  en una GPU de 8 GB de VRAM.
- Evaluación cuantitativa con Recall@k, Precision@k, nDCG@k y MRR, comparando el modelo
  original contra el fine-tuned sobre el mismo conjunto de validación.
- Base vectorial ChromaDB con las 44 471 películas indexadas.
- API REST (FastAPI) con búsqueda semántica en tiempo real.
- Frontend web sin dependencias ni build step.

## Tecnologías

Python 3.13 · PyTorch (CUDA) · SentenceTransformers · Transformers · Accelerate ·
Datasets (Hugging Face) · scikit-learn · pandas / numpy · ChromaDB · FastAPI · Uvicorn ·
Matplotlib · TensorBoard · HTML/CSS/JS

## Requisitos

- Python 3.13+
- GPU NVIDIA con CUDA (solo necesaria para re-entrenar; **no** hace falta para servir la
  API con el modelo ya entrenado)
- Credenciales de Kaggle, solo si se va a descargar el dataset desde cero
  (`~/.kaggle/kaggle.json` o variables `KAGGLE_USERNAME` / `KAGGLE_KEY`)

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r scripts\requirements.txt
```

`torch` con soporte CUDA se instala aparte según la GPU disponible (ver
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)) — no se reinstala
automáticamente con `requirements.txt` para no pisar una build con CUDA ya funcionando.

Nota: la API (`fastapi`, `uvicorn`, `chromadb`) puede vivir en el mismo entorno que el
entrenamiento o en uno separado. Ver detalle en `DOCUMENTACION_TECNICA.md`, sección 11.

## Llevar el proyecto a otro equipo

El repositorio en git solo versiona código (`.gitignore` excluye todo lo pesado y
regenerable: `.venv/`, `models/`, `clases_sr/chroma_db/`,
`reports/_chroma_comparacion_original/`, `data/processed/`, `data/training/`). Al clonar
en otro equipo, esas carpetas **no existen** y hay dos formas de conseguirlas:

**Opción A — copiar los artefactos ya generados (recomendado, no requiere GPU ni Kaggle):**
Copiar manualmente (USB, disco compartido, almacenamiento en la nube) estas carpetas desde
el equipo original, respetando la misma ubicación relativa a la raíz del proyecto:

| Carpeta | Peso aprox. | Necesaria para |
|---|---|---|
| `models/bge-m3-finetuned/` | 2.2 GB | Servir la API (`main.py`) en modo local |
| `clases_sr/chroma_db/` | 406 MB | Servir la API (`main.py`), ambos modos |
| `data/processed/` | 240 MB | Reconstruir la base vectorial (`cargar.py`) o regenerar `data/training/` |
| `reports/_chroma_comparacion_original/` | 370 MB | Solo si se va a usar el modo "online" (comparación contra el modelo original) |
| `data/training/` | 60 MB | Solo si se va a re-entrenar o auditar el dataset de entrenamiento |
| `models/_checkpoints/` | 4.3 GB | Solo si se va a reanudar un entrenamiento interrumpido (no hace falta para servir la API) |

Con solo `models/bge-m3-finetuned/`, `clases_sr/chroma_db/` y el código, la API ya
funciona en modo local sin tocar el resto del pipeline.

**Opción B — regenerar todo desde cero corriendo el pipeline** (ver "Cómo ejecutar" más
abajo). Requiere credenciales de Kaggle, una GPU con CUDA para que el fine-tuning sea
viable en tiempo razonable, y varias horas.

Recordar también la separación de entornos documentada arriba: el entrenamiento
(`scripts/`) y la API (`clases_sr/`) pueden requerir dos entornos Python distintos según
cómo se instalen las dependencias — no asumir que un solo `pip install -r requirements.txt`
alcanza para ambos si se optó por separarlos.

## Configuración

Todos los hiperparámetros del fine-tuning están centralizados en [`config.py`](./config.py)
(rutas, split train/validación, tamaño de batch, learning rate, optimizador, precisión,
early stopping, valores de k para la evaluación). No requiere variables de entorno para
correr con los valores por defecto.

## Cómo ejecutar

```bash
# 1. Preparar el dataset
python scripts\preparar_dataset.py

# 2. Generar pares y tripletas de entrenamiento
python scripts\generar_dataset_entrenamiento.py

# 3. Validar y enriquecer con Hard Negative Mining
python scripts\validar_dataset_entrenamiento.py

# 4. Fine-tuning
python scripts\finetuning.py

# 5. Construir la base vectorial con el modelo fine-tuned
python clases_sr\cargar.py

# 6. Levantar la API
cd clases_sr
uvicorn main:app --reload
```

Luego abrir `clases_sr/index.html` en el navegador (con la API corriendo en
`http://127.0.0.1:8000`).

Guía paso a paso completa, con detalle de cada script: ver
[`DOCUMENTACION_TECNICA.md`](./DOCUMENTACION_TECNICA.md), secciones 12 y 13.

## Capturas

> _Espacio reservado para capturas de pantalla del frontend y de la API en uso._

| Frontend — búsqueda | Resultados |
|---|---|
| _(agregar captura)_ | _(agregar captura)_ |

| TensorBoard — curva de entrenamiento |
|---|
| _(agregar captura, o usar `logs/graficos/training_curve.png`)_ |

## Resultados obtenidos

El modelo fine-tuned mejoró **11 de 11 métricas** de recuperación medidas sobre el
conjunto de validación, frente al modelo `BAAI/bge-m3` original.

| Métrica | Original | Fine-Tuned | Mejora |
|---|---|---|---|
| Recall@1 | 0.1756 | 0.3269 | +86.1% |
| Recall@5 | 0.3218 | 0.6423 | +99.6% |
| Recall@10 | 0.4218 | 0.7885 | +86.9% |
| MRR | 0.2633 | 0.4710 | +78.9% |
| nDCG@10 | 0.2876 | 0.5364 | +86.5% |

Detalle completo de todas las métricas, interpretación y comparación por consulta real:
ver [`RESULTADOS.md`](./RESULTADOS.md).

## Métricas

Evaluadas sobre el conjunto de validación (400 tripletas, split 90/10 por película única,
sin fuga de información entre train y validación): Recall@{1,5,10}, Precision@{1,5,10},
nDCG@{1,5,10}, MRR y similitud coseno promedio ancla-positivo.

## Estructura del proyecto

```
scripts/              → pipeline offline: dataset, pares/tripletas, validación, fine-tuning
clases_sr/             → API (FastAPI), indexado ChromaDB, frontend
config.py              → configuración central del fine-tuning
data/processed/         → dataset limpio (44 471 películas)
data/training/          → pares, tripletas y datasets de entrenamiento (v1 y v2)
data/reports/            → reporte y gráficos de validación del dataset
models/bge-m3-finetuned/ → modelo final, listo para usar
models/_checkpoints/      → checkpoints intermedios del entrenamiento
logs/                     → historial, métricas y gráficos del entrenamiento
reports/                   → comparación final: modelo original vs. fine-tuned
```

Detalle carpeta por carpeta y script por script:
[`DOCUMENTACION_TECNICA.md`](./DOCUMENTACION_TECNICA.md), secciones 2 y 3.

## Licencia

Proyecto académico / de portafolio. El dataset ("The Movies Dataset", Kaggle) y el
modelo base (`BAAI/bge-m3`) mantienen sus licencias originales; consultar
[Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) y
[Hugging Face](https://huggingface.co/BAAI/bge-m3) respectivamente.

## Trabajo futuro

- Entrenamiento definitivo (sin los límites de "piloto" de `config.py`: dataset completo,
  más épocas) — ver guía en `DOCUMENTACION_TECNICA.md`, sección 13.
- Evaluación con métricas humanas (no solo automáticas) sobre una muestra de consultas
  reales.
- Filtros adicionales en la API (por género, año, rango de similitud mínima).
- Separar los scripts de prueba/exploración del curso del código de producción dentro
  de `clases_sr/` (ver `AUDITORIA.md`).
