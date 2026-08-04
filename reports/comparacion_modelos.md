# Comparación: Modelo Original vs. Modelo Fine-Tuned

Generado: 2026-08-03 00:14:18

## Resumen del entrenamiento

- Modelo base: `BAAI/bge-m3`
- Fecha de entrenamiento: 2026-08-02T23:19:53
- GPU: NVIDIA GeForce RTX 4060 Ti (8.0 GB VRAM)
- PyTorch 2.11.0+cu128 / CUDA 12.8
- Épocas: 1 | Batch: 32 | Learning rate: 2e-05
- Ejemplos de entrenamiento (piloto): 2000 | validación: 400

## Métricas obtenidas (conjunto de validación)

| Métrica | Original | Fine-Tuned | Mejora |
|---|---|---|---|
| recall@1 | 0.1756 | 0.3269 | +86.1% |
| recall@5 | 0.3218 | 0.6423 | +99.6% |
| recall@10 | 0.4218 | 0.7885 | +86.9% |
| precision@1 | 0.1769 | 0.3308 | +87.0% |
| precision@5 | 0.0656 | 0.1297 | +97.7% |
| precision@10 | 0.0428 | 0.0795 | +85.6% |
| ndcg@1 | 0.1769 | 0.3308 | +87.0% |
| ndcg@5 | 0.2560 | 0.4889 | +91.0% |
| ndcg@10 | 0.2876 | 0.5364 | +86.5% |
| mrr | 0.2633 | 0.4710 | +78.9% |
| cosine_similarity_promedio | 0.5926 | 0.5949 | +0.4% |

## Verificación de la API con consultas reales

Consultas probadas contra `/buscar`: astronautas, piratas, terror psicológico, robots, amor imposible, viajes en el tiempo.
Todas respondieron correctamente (ver detalle de la comparación abajo).

## Comparación por consulta: Original vs. Fine-Tuned

### "astronautas"

- Tiempo de respuesta: Original 0.3965s | Fine-Tuned 0.5725s

| # | Original (título — categoría — sim) | Fine-Tuned (título — categoría — sim) |
|---|---|---|
| 1 | Countdown — Thriller, Science Fiction — 0.5764 | In the Shadow of the Moon — Documentary — 0.4713 |
| 2 | A Space Program — Drama, Documentary, Adventure — 0.5588 | The Stranger — TV Movie, Science Fiction — 0.4261 |
| 3 | Space Cowboys — Action, Adventure, Thriller — 0.5565 | For All Mankind — Documentary — 0.4147 |
| 4 | The American Astronaut — Action, Comedy, Science Fiction, Music — 0.5549 | Space Station 3D — Documentary — 0.4079 |
| 5 | The Aftermath — Action, Adventure, Science Fiction, Horror — 0.5541 | A Space Program — Drama, Documentary, Adventure — 0.4056 |

### "piratas"

- Tiempo de respuesta: Original 0.0291s | Fine-Tuned 0.0269s

| # | Original (título — categoría — sim) | Fine-Tuned (título — categoría — sim) |
|---|---|---|
| 1 | Pirates — Adventure, Comedy — 0.6005 | Tales of the Black Freighter — Animation, Horror, Action — 0.4078 |
| 2 | The Pirates! In an Adventure with Scientists! — Animation, Adventure, Family, Comedy — 0.5847 | The Golden Hawk — Adventure — 0.4028 |
| 3 | The Pirate Movie — Action, Comedy, Drama, Family, Music, Romance — 0.5623 | The Boy and the Pirates — Fantasy, Family, Adventure — 0.3949 |
| 4 | City of Pirates — Drama, Fantasy, Horror, Thriller — 0.5595 | Blackie the Pirate — Action, Adventure, Comedy — 0.3918 |
| 5 | The Crimson Pirate — Action, Adventure, Comedy — 0.5569 | The Crimson Pirate — Action, Adventure, Comedy — 0.3884 |

### "terror psicológico"

- Tiempo de respuesta: Original 0.0234s | Fine-Tuned 0.024s

| # | Original (título — categoría — sim) | Fine-Tuned (título — categoría — sim) |
|---|---|---|
| 1 | Psychosis — Horror, Mystery, Thriller — 0.5879 | Boogeyman 2 — Horror, Thriller — 0.4867 |
| 2 | Devil Times Five — Horror — 0.5814 | Fright — Horror, Thriller — 0.4805 |
| 3 | Christmas Evil — Drama, Horror — 0.58 | Black Christmas — Horror, Thriller — 0.473 |
| 4 | Psycho — Horror, Mystery, Thriller — 0.5751 | Paranormal Activity: The Ghost Dimension — Horror, Thriller — 0.4656 |
| 5 | A Fantastic Fear of Everything — Comedy, Thriller — 0.5722 | Bad Karma — Thriller, Horror — 0.4654 |

### "robots"

- Tiempo de respuesta: Original 0.0272s | Fine-Tuned 0.0191s

| # | Original (título — categoría — sim) | Fine-Tuned (título — categoría — sim) |
|---|---|---|
| 1 | Robots — Animation, Comedy, Family, Science Fiction — 0.6504 | The Transformers: The Movie — Animation — 0.4287 |
| 2 | Robot Holocaust — Science Fiction — 0.5984 | The Invisible Boy — Science Fiction — 0.4193 |
| 3 | I, Robot — Action, Science Fiction — 0.591 | Hot Bot — Comedy, Science Fiction — 0.4072 |
| 4 | Robot Overlords — Adventure, Science Fiction, Action — 0.5888 | Enthiran — Action, Comedy, Drama, Romance, Science Fiction — 0.404 |
| 5 | Robot Stories — Drama, Science Fiction, Romance — 0.5865 | Robot Monster — Science Fiction — 0.4023 |

### "amor imposible"

- Tiempo de respuesta: Original 0.0208s | Fine-Tuned 0.0215s

| # | Original (título — categoría — sim) | Fine-Tuned (título — categoría — sim) |
|---|---|---|
| 1 | Pyaar Impossible — Drama, Comedy, Romance, Foreign — 0.6318 | Pierre et Djemila —  — 0.4002 |
| 2 | Amor Impossível — Drama — 0.6032 | Aashiq —  — 0.3897 |
| 3 | Inevitable — Drama — 0.5606 | Act Of Love —  — 0.3693 |
| 4 | Albela —  — 0.5507 | DOS μια ιστορία αγάπης, απ' την ανάποδη —  — 0.3661 |
| 5 | Act Of Love —  — 0.5449 | Saaya —  — 0.3659 |

### "viajes en el tiempo"

- Tiempo de respuesta: Original 0.0281s | Fine-Tuned 0.0193s

| # | Original (título — categoría — sim) | Fine-Tuned (título — categoría — sim) |
|---|---|---|
| 1 | Timecrimes — Science Fiction, Thriller — 0.6065 | 12:01 PM — Science Fiction — 0.4516 |
| 2 | Crusade in Jeans — Adventure, Fantasy, History — 0.6045 | Rewind — Science Fiction — 0.4442 |
| 3 | Voyage in Time — Documentary, Foreign — 0.6024 | Dr. Who and the Daleks — Science Fiction, Adventure — 0.4285 |
| 4 | The Time Travelers — Science Fiction — 0.6011 | Daleks' Invasion Earth: 2150 A.D. — Science Fiction — 0.4277 |
| 5 | Time Bandits — Family, Fantasy, Science Fiction, Adventure, Comedy — 0.5976 | Crusade in Jeans — Adventure, Fantasy, History — 0.4265 |

## Conclusión

El modelo Fine-Tuned mejoró 11 de 11 métricas de recuperación medidas sobre el conjunto de validación (Recall@1 subió de 0.176 a 0.327, MRR de 0.263 a 0.471).

En las 6 consultas reales de prueba, la similitud coseno promedio de los resultados fue 0.5827 con el modelo original y 0.4205 con el Fine-Tuned (similitud menor en este puñado de consultas puntuales; la mejora medida en el conjunto de validación, con más ejemplos, es la referencia más confiable).

Conclusión: la migración al modelo Fine-Tuned mejora la calidad de recuperación del recomendador respecto al modelo BAAI/bge-m3 original, sin cambios en la API ni en el dataset.
