# Informe de Resultados

Fuente de datos: `logs/evaluacion_comparativa.json`, `logs/training_config.json`,
`reports/comparacion_modelos.md` (generado 2026-08-03 00:14:18).

---

## 1. Condiciones de la evaluación

- **Modelo original:** `BAAI/bge-m3`, sin fine-tuning.
- **Modelo fine-tuned:** `BAAI/bge-m3` + fine-tuning piloto (`models/bge-m3-finetuned/`).
- **Conjunto de evaluación:** 400 tripletas de validación (10% de las anclas únicas,
  nunca vistas durante el entrenamiento — split hecho por película, no por fila).
- **Corpus de recuperación:** positivos + negativos del split de validación.
- **Hardware:** NVIDIA GeForce RTX 4060 Ti (8 GB VRAM), PyTorch 2.11.0+cu128, CUDA 12.8.
- **Entrenamiento:** 1 época, 2000 ejemplos de entrenamiento (piloto), batch efectivo 32,
  learning rate 2e-5, optimizador Adafactor, FP16, `max_seq_length` 256.

## 2. Métricas antes del Fine-Tuning (modelo original)

| Métrica | Valor |
|---|---|
| Recall@1 | 0.1756 |
| Recall@5 | 0.3218 |
| Recall@10 | 0.4218 |
| Precision@1 | 0.1769 |
| Precision@5 | 0.0656 |
| Precision@10 | 0.0428 |
| nDCG@1 | 0.1769 |
| nDCG@5 | 0.2560 |
| nDCG@10 | 0.2876 |
| MRR | 0.2633 |
| Similitud coseno promedio (ancla-positivo) | 0.5926 |

## 3. Métricas después del Fine-Tuning (modelo fine-tuned)

| Métrica | Valor |
|---|---|
| Recall@1 | 0.3269 |
| Recall@5 | 0.6423 |
| Recall@10 | 0.7885 |
| Precision@1 | 0.3308 |
| Precision@5 | 0.1297 |
| Precision@10 | 0.0795 |
| nDCG@1 | 0.3308 |
| nDCG@5 | 0.4889 |
| nDCG@10 | 0.5364 |
| MRR | 0.4710 |
| Similitud coseno promedio (ancla-positivo) | 0.5949 |

## 4. Tabla comparativa

| Métrica | Original | Fine-Tuned | Cambio absoluto | Mejora relativa |
|---|---|---|---|---|
| Recall@1 | 0.1756 | 0.3269 | +0.1513 | **+86.1%** |
| Recall@5 | 0.3218 | 0.6423 | +0.3205 | **+99.6%** |
| Recall@10 | 0.4218 | 0.7885 | +0.3667 | **+86.9%** |
| Precision@1 | 0.1769 | 0.3308 | +0.1539 | **+87.0%** |
| Precision@5 | 0.0656 | 0.1297 | +0.0641 | **+97.7%** |
| Precision@10 | 0.0428 | 0.0795 | +0.0367 | **+85.6%** |
| nDCG@1 | 0.1769 | 0.3308 | +0.1539 | **+87.0%** |
| nDCG@5 | 0.2560 | 0.4889 | +0.2329 | **+91.0%** |
| nDCG@10 | 0.2876 | 0.5364 | +0.2488 | **+86.5%** |
| MRR | 0.2633 | 0.4710 | +0.2077 | **+78.9%** |
| Similitud coseno promedio | 0.5926 | 0.5949 | +0.0023 | +0.4% |

**Resultado: mejora en 11 de 11 métricas medidas.**

## 5. Interpretación de cada métrica

- **Recall@k** — de todos los documentos realmente relevantes para una consulta, qué
  fracción aparece entre los primeros *k* resultados devueltos. Subió de forma muy
  marcada en las tres ventanas medidas (k=1, 5, 10), lo que indica que el modelo
  fine-tuned no solo acierta más seguido, sino que deja de "perder" resultados
  relevantes fuera del top.

- **Precision@k** — de los *k* resultados devueltos, qué fracción es efectivamente
  relevante. También mejoró de forma consistente: el modelo fine-tuned devuelve listas
  de resultados más "limpias", con menos ruido irrelevante mezclado.

- **nDCG@k (normalized Discounted Cumulative Gain)** — mide no solo si lo relevante
  aparece, sino si aparece *arriba* en el ranking (penaliza que un resultado relevante
  esté en la posición 9 en vez de la 1). La mejora de +86% a +91% en nDCG indica que el
  fine-tuning no solo trae más resultados correctos, sino que los ordena mejor.

- **MRR (Mean Reciprocal Rank)** — promedio de 1/posición del primer resultado
  relevante. Subir de 0.263 a 0.471 significa que, en promedio, el primer resultado
  correcto pasó de aparecer cerca de la posición 4 a aparecer cerca de la posición 2.

- **Similitud coseno promedio (ancla-positivo)** — mide qué tan cerca quedan, en el
  espacio de embeddings, una película y su par positivo real. Es la métrica que menos
  se movió (+0.4%): esperable, porque el objetivo del entrenamiento
  (`MultipleNegativesRankingLoss`) optimiza el *orden relativo* entre positivos y
  negativos, no la magnitud absoluta de la similitud. Por eso Recall/Precision/nDCG/MRR
  —que sí miden ordenamiento— son las métricas que reflejan la mejora real.

## 6. Verificación adicional: consultas reales contra la API

Además de la evaluación cuantitativa sobre el conjunto de validación,
`scripts/verificar_migracion.py` corrió 6 consultas reales contra la API en producción
(`astronautas`, `piratas`, `terror psicológico`, `robots`, `amor imposible`, `viajes en
el tiempo`), comparando los resultados del modelo original y el fine-tuned sobre el
corpus completo de 44 471 películas (no solo el conjunto de validación).

En ese conjunto puntual de 6 consultas, la similitud coseno promedio de los resultados
fue 0.5827 (original) vs. 0.4205 (fine-tuned) — más baja con el fine-tuned. Esto **no
contradice** la mejora medida arriba: son solo 6 consultas manuales (una muestra
pequeña y no aleatoria) contra las 400 tripletas del conjunto de validación real. La
métrica de referencia para juzgar la calidad del modelo es la de la sección 4, no esta
verificación puntual — que sirvió, sobre todo, para confirmar que el sistema completo
(API + ChromaDB + modelo) funciona end-to-end con datos reales, no para medir calidad.
Detalle completo por consulta en `reports/comparacion_modelos.md`.

## 7. Contexto del dataset de entrenamiento

| Estadística | Valor |
|---|---|
| Películas totales | 44 471 |
| Tripletas totales (fáciles + hard) | 23 857 |
| Tripletas con hard negative | 3 643 |
| Similitud TF-IDF promedio ancla-positivo | 0.0913 |
| Similitud TF-IDF promedio ancla-negativo (fácil) | 0.0092 |
| Similitud TF-IDF promedio ancla-negativo (hard) | 0.1559 |
| Brecha positivo-negativo (fácil) | 0.0821 |
| Brecha positivo-negativo (hard) | 0.0486 |

La brecha entre positivo y negativo es, por construcción, siempre positiva pero mucho
más chica en los hard negatives (0.0486 vs. 0.0821): son, literalmente, los ejemplos
más difíciles de distinguir, y su inclusión en el entrenamiento (etapa 3, Hard Negative
Mining) es lo que le exige al modelo aprender límites de decisión más finos que los que
aprendería solo con negativos fáciles.

## 8. Conclusiones

1. El fine-tuning produjo una mejora **grande y consistente** en las métricas de
   recuperación relevantes (Recall, Precision, nDCG, MRR), no una mejora marginal o
   dentro del margen de ruido.
2. La mejora se sostiene en las tres ventanas de corte evaluadas (k=1, 5, 10), lo que
   indica una mejora robusta del ranking en general, no un efecto aislado en un solo
   punto de corte.
3. Este resultado se obtuvo con un entrenamiento **piloto** (1 época, subconjunto del
   dataset) — es evidencia suficiente de que el enfoque funciona y justifica invertir en
   el entrenamiento definitivo con el dataset completo (ver `DOCUMENTACION_TECNICA.md`,
   sección 13) para exprimir mejoras adicionales.
4. El sistema fue verificado end-to-end (API real + ChromaDB con las 44 471 películas +
   consultas reales), no solo evaluado offline: el proyecto cumple su objetivo tanto en
   las métricas como en el sistema funcionando.
