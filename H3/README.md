# H3 — Recomendación multimodal de videojuegos en Steam

Código y cuadernillos ejecutados de la Hito 3 del proyecto (IIC3633). El foco es
**reproducir y extender CPGRec/CPGRec+** (un recomendador GNN multimodal para Steam)
frente a un conjunto de _baselines_ colaborativos y de contenido, evaluados sobre
**tres datasets** de Steam y bajo **dos protocolos de evaluación**. El hallazgo
central del trabajo es metodológico: **el protocolo de evaluación decide quién gana**
— un mismo modelo pasa de "estado del arte" a "peor que popularidad" según cómo se
mida.

Todos los `.ipynb` se entregan **ya ejecutados** (conservan sus salidas): abrirlos
permite ver los resultados sin re-correr nada. Para replicar desde cero, ver la
sección [§7](#7-cómo-replicar).

---

## 1. Estructura de archivos

```
H3/
├── README.md
├── recsys_protocol.py          # Módulo común: seed, k-core, splits, métricas
├── generar_figuras.py          # Genera las figuras del informe (figs/*.pdf)
│
│  ── Datos ─────────────────────────────────────────────────────────────
├── load_scgrec.ipynb           # Descarga y arma el dataset SCGRec (parquets)
├── profile_datasets.ipynb      # Estadísticas descriptivas de los 3 datasets
│
│  ── F2 · Comparativo CPGRec (GNN) — tabla principal ────────────────────
├── CPGRec_comparativo_multiseed_T2_corregido.ipynb        # SCGRec
├── CPGRec_comparativo_multiseed_kozyriev_corregido.ipynb  # Kozyriev
├── CPGRec_comparativo_multiseed_ucsd_corregido.ipynb      # UCSD  (+ FM/DeepFM/DeepNN + CB-SBERT)
│
│  ── F2 · Baselines ─────────────────────────────────────────────────────
├── bpr_scgrec_h3.ipynb         # BPR — SCGRec
├── bpr_kozyriev_h3.ipynb       # BPR — Kozyriev
├── bpr_ucsd_h3.ipynb           # BPR — UCSD
├── sbert_kozyriev_h3.ipynb     # CB-SBERT (contenido) — Kozyriev
├── sbert_ucsd_h3.ipynb         # CB-SBERT (contenido) — UCSD
├── deep_kozyriev_h3.ipynb      # FM/DeepFM/DeepNN (full-ranking) — Kozyriev
│
│  ── F3 · Reproducción Cheuque (protocolo per-user) ─────────────────────
├── cheuque_ucsd_inicial.ipynb  # UCSD — filtro denso + re-ranking per-user (leaky)
├── cheuque_kozyriev.ipynb      # Kozyriev — idem
│
│  ── Reproducción / diagnóstico SCGRec (metodológico) ───────────────────
├── CPGRec_repro_scgrec_T2.ipynb # Validación de la reproducción CPGRec vs. el paper
└── diag_train_T1.ipynb          # Diagnóstico: efecto de épocas y de freeze_user_llm
```

---

## 2. Datasets

| Dataset                                                       | Señal                                             | Crudo                                   | Tras filtro                        | Rol                                |
| ------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------- | ---------------------------------- | ---------------------------------- |
| **UCSD / McAuley** (Australian Steam)                         | _ownership_ + `playtime_forever`                  | 70.912 u · 10.978 j · 5,15 M interac.   | 5-core (5,5): 62.944 u · 9.198 j   | Denso; dataset de Cheuque'19       |
| **Kozyriev** (`game-recommendations-on-steam`) + FronkonGames | reseñas (`is_recommended`, `hours`)               | 13,78 M u · 37.610 j · 41,15 M interac. | k-core (5,20): 1,90 M u · 22.676 j | Disperso; catálogo grande          |
| **SCGRec**                                                    | interacciones + categorías (género/dev/publisher) | 3,9 M u · 2.675 j · 95,2 M interac.     | _split_ oficial 80/10/10           | Dataset original de CPGRec/CPGRec+ |

Notas de datos:

- `hours` de Kozyriev ya viene **en horas**; `playtime_forever` de UCSD viene **en minutos**.
- Kozyriev no trae géneros → se completan con **FronkonGames** (`Genres`, `Metacritic`).
- SCGRec trae un **split oficial 80/10/10** (se respeta para la reproducción).
- `load_scgrec.ipynb` deja los `.parquet` de SCGRec (`inter`, `game_categories`).
  Los cuadernillos de UCSD y Kozyriev incluyen su propio _bootstrap_: descargan y
  parsean el dataset si faltan los `.parquet` (no requieren preparar archivos a mano).
- Fuentes: UCSD (McAuley Lab, cseweb.ucsd.edu), Kozyriev (Kaggle) + FronkonGames
  (Kaggle), SCGRec (repositorio de los autores).

---

## 3. Regímenes de evaluación

El proyecto usa **dos regímenes** que **no son intercambiables** (distinto filtro,
_split_ y métricas). Pegar números de uno en la tabla del otro sería una comparación
injusta; por eso viven en cuadernillos separados.

### F2 — protocolo común, honesto _(tabla principal)_

- **Filtro:** k-core (5,20); UCSD 5-core (5,5); SCGRec usa su _split_ oficial.
- **Split:** aleatorio **80/10/10** por interacción (SCGRec: oficial).
- **Evaluación:** **full-ranking** sobre **todo el catálogo** (se excluyen los ítems
  ya vistos en _train_).
- **Métricas:** Recall / NDCG / Hit / Precision @{5,10} (multi-relevante) + Coverage,
  _genre-coverage_, entropía y desglose _long-tail_ por nivel de actividad del usuario.

### F3 — protocolo per-user à la Cheuque'19 *(robustez / *stress test*)*

- **Filtro denso:** `iterative_filter(min_user=100, min_item=200)`
  (Kozyriev colapsa a 0 con (100,200) → _fallback_ al subset viable **(50,100)**).
- **Split:** por usuario (`split_per_user`, 80/20).
- **Evaluación:** **re-ranking per-user** (rankear solo entre los ítems de _test_ del
  propio usuario) — este es el protocolo que **infla** las métricas.
- **Dos columnas:** _con fuga_ (usa `playtime` como _feature_, que define el _label_
  `playtime ≥ 5 h`) y _sin fuga_. ALS full-ranking se corre como contraste honesto.
- **Métricas:** NDCG@10 / MAP@10.

> **Por qué F3 importa:** los mismos modelos _deep_ dan NDCG@10 ≈ **0,99** en F3 y
> ≈ **0,02–0,14** en F2. Ese 0,99 es un artefacto de protocolo en **dos capas**:
> la fuga de `playtime` (0,99 → ~0,71–0,91) y, sobre todo, el _re-ranking_ per-user
> vs. el full-ranking real (~0,71 → ~0,02–0,14).

---

## 4. Configuración común (protocolo, seeds, hiperparámetros)

**Semillas.** Semilla global = **42** (fijada por `recsys_protocol.set_global_seed`).

- **CPGRec / GNN (F2):** **3 semillas** `[42, 1, 2]` en una sola corrida → se reporta
  **media ± d.e.** (los _deltas_ entre variantes son del orden del ruido, así que la
  desviación _es_ parte del resultado).
- **BPR, CB-SBERT, FM/DeepFM/DeepNN (F2):** **1 semilla** (42). El _full-ranking_ de
  estos modelos es caro y su señal es robusta; MostPop/ALS reproducen _byte a byte_
  los del comparativo GNN, lo que sirve de _self-check_.
- **Cheuque (F3):** semilla 42; 3 _folds_ (submuestreo repetido).

**Convención de `playtime`.** Canónica = **horas** (`/60`) + **dedup** de pares
`(user, app)`, consistente entre los 3 datasets. Los comparativos GNN de UCSD se
publicaron en **minutos + sin dedup**; esa convención **solo mueve a ALS**
(Recall@5 0,087 → 0,105), porque el resto de modelos no usa `playtime` como _feature_.

**Filtro de usuarios (submuestra).** Para que la corrida quepa en una sesión de GPU:
SCGRec **2 M** usuarios, Kozyriev **500 K**, UCSD **sin submuestra** (~63 K).
La evaluación _long-tail_ se hace sobre `N_EVAL = 200.000` usuarios (o menos).

**Hiperparámetros por modelo:**

| Modelo                   | Configuración                                                                                                                                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CPGRec (GNN)**         | `emb=32`, `lr=0,03`, `batch=1024`, `m=6,5` (reponderación de negativos del NSR), `β=0,1`, `γ=80`, **1000 épocas**, `freeze_user_llm=0` (la 2.ª tabla de _embeddings_ de usuario es entrenable, como el repo original). Variantes: **base** (`--use_per 0 --use_llm 0`) · **+PER** (`--use_per 1`) · **+PER+PRG** (`--use_per 1 --use_llm 1`). |
| **+PER**                 | Pesos de preferencia derivados del _rating_ del juego (metascore en SCGRec; `positive_ratio` en Kozyriev/UCSD) vía test F. Depende de la cobertura del _rating_.                                                                                                                                                      |
| **+PRG-SBERT**           | _Embeddings_ de contenido `all-MiniLM-L6-v2` (384-dim) reducidos con **PCA a 64** (evita OOM del `user_embedding_LLM` a gran escala).                                                                                                                                                                                 |
| **ALS**                  | `factors=64`, `iters=15`, `reg=0,1`; confianza `1 + 40·log1p(horas)` (`implicit`).                                                                                                                                                                                                                                    |
| **BPR**                  | `factors=64`, `lr=0,01`, `reg=0,01`, `iters=100` (`implicit`).                                                                                                                                                                                                                                                        |
| **CB-SBERT**             | Recomendador de contenido puro: `all-MiniLM-L6-v2` sobre texto del juego; ranking por producto punto (en CPU).                                                                                                                                                                                                        |
| **FM / DeepFM / DeepNN** | `deepctr-torch`; `emb=32`, _negative sampling_ 4:1, pérdida BCE, 10 épocas, `batch=16384`, evaluados en **full-ranking** (sin fuga de `playtime`).                                                                                                                                                                    |
| **MostPop**              | Popularidad global (piso / _sanity check_).                                                                                                                                                                                                                                                                           |

**Modelo GNN.** Los cuadernillos CPGRec clonan el repositorio del modelo
(`HsipingLi/CPGRec-Plus`) y sobre él escriben los _builders_ de grafo, el _dataloader_
y el _driver_ de entrenamiento. La red se ejecuta en GPU; la evaluación
(_embeddings_ → producto punto) es liviana.

---

## 5. Qué produce cada cuadernillo

| Cuadernillo                                       | Régimen | Dataset  | Modelos / salida                                                       | Filtro                      | Split    | Seeds   |
| ------------------------------------------------- | ------- | -------- | ---------------------------------------------------------------------- | --------------------------- | -------- | ------- |
| `CPGRec_comparativo_multiseed_T2_corregido`       | F2      | SCGRec   | MostPop, ALS, **CPGRec** base/+PER/+PER+PRG                            | _split_ oficial (sub 2 M u) | 80/10/10 | 3       |
| `CPGRec_comparativo_multiseed_kozyriev_corregido` | F2      | Kozyriev | idem                                                                   | k-core (5,20), sub 500 K    | 80/10/10 | 3       |
| `CPGRec_comparativo_multiseed_ucsd_corregido`     | F2      | UCSD     | idem **+ FM/DeepFM/DeepNN + CB-SBERT**                                 | 5-core (5,5)                | 80/10/10 | 3 (GNN) |
| `bpr_scgrec_h3`                                   | F2      | SCGRec   | **BPR** (+ MostPop/ALS _self-check_)                                   | sub 2 M u                   | 80/10/10 | 1       |
| `bpr_kozyriev_h3`                                 | F2      | Kozyriev | **BPR**                                                                | k-core (5,20), sub 500 K    | 80/10/10 | 1       |
| `bpr_ucsd_h3`                                     | F2      | UCSD     | **BPR**                                                                | 5-core (5,5)                | 80/10/10 | 1       |
| `sbert_kozyriev_h3`                               | F2      | Kozyriev | **CB-SBERT**                                                           | k-core (5,20), sub 500 K    | 80/10/10 | 1       |
| `sbert_ucsd_h3`                                   | F2      | UCSD     | **CB-SBERT**                                                           | 5-core (5,5)                | 80/10/10 | 1       |
| `deep_kozyriev_h3`                                | F2      | Kozyriev | **FM/DeepFM/DeepNN** (full-ranking)                                    | k-core (5,20), sub 500 K    | 80/10/10 | 1       |
| `cheuque_ucsd_inicial`                            | F3      | UCSD     | ALS + FM/DeepFM/DeepNN, con/sin fuga                                   | denso (100,200)             | per-user | 3 folds |
| `cheuque_kozyriev`                                | F3      | Kozyriev | idem                                                                   | denso (50,100)              | per-user | 3 folds |
| `CPGRec_repro_scgrec_T2`                          | —       | SCGRec   | Reproducción CPGRec base/+PRG vs. el paper (300 ép., régimen inicial)  | _split_ oficial             | 80/10/10 | 1       |
| `diag_train_T1`                                   | —       | SCGRec   | Aísla el efecto de **épocas (300 vs 1000)** y de **`freeze_user_llm`** | _split_ oficial (200 K u)   | 80/10/10 | —       |
| `load_scgrec`                                     | —       | SCGRec   | Descarga/parsea SCGRec → `.parquet`                                    | —                           | —        | —       |
| `profile_datasets`                                | —       | los 3    | Estadísticas descriptivas (crudo, k-core, densidad)                    | —                           | —        | —       |

Cada cuadernillo F2 **imprime todas sus tablas en `stdout`** (accuracy, diversidad,
_long-tail_, ejemplos con nombres de juegos) y las persiste como CSV/JSON, de modo que
los resultados quedan visibles aunque no se descarguen archivos.

---

## 6. Requisitos y entorno

Pensado para ejecutarse en **Google Colab** (GPU). Dependencias principales:

- **Python 3.11**, `numpy`, `pandas`, `scikit-learn`
- **`torch 2.4` + `dgl 2.4` (cu118)** — solo para los cuadernillos CPGRec (GNN)
- **`implicit`** — ALS y BPR
- **`sentence-transformers`** (`all-MiniLM-L6-v2`) — CB-SBERT y la rama +PRG
- **`deepctr-torch`** — FM/DeepFM/DeepNN
- **`kagglehub` / `gdown`** — descarga de datasets

> **Gotcha de Colab (cuadernillos GNN):** la celda de _setup_ instala `torch 2.4`
>
> - DGL y **reinicia el runtime la primera vez**. Flujo correcto: correr esa celda
>   **sola**, esperar a que imprima `dgl ... OK` (sin reinicio en la 2.ª pasada), y
>   recién entonces **"Ejecutar todo" desde arriba**. Si se hace "Ejecutar todo" en frío,
>   el reinicio corta la corrida a mitad.

---

## 7. Cómo replicar

Orden sugerido:

1. **Datos** — `load_scgrec.ipynb` (deja los `.parquet` de SCGRec). Los cuadernillos de
   UCSD/Kozyriev se auto-abastecen (bootstrap). `profile_datasets.ipynb` es opcional
   (solo estadísticas descriptivas).
2. **F2 — tabla principal** — los tres `CPGRec_comparativo_multiseed_*_corregido`,
   más `bpr_*`, `sbert_*` y `deep_kozyriev_h3`.
3. **F3 — robustez** — `cheuque_ucsd_inicial` y `cheuque_kozyriev`.
4. **Figuras** — `python generar_figuras.py` → escribe `figs/*.pdf` (usa cifras ya
   consolidadas de las corridas anteriores; no requiere GPU).

Cada cuadernillo trae una variable `TIER`: `'T1'` = _sanity_ rápido (menos usuarios,
menos épocas, evaluación acotada) y `'T2'` = corrida final. Se recomienda correr `T1`
antes que `T2` para validar la fontanería.

### GPU recomendada por tipo de corrida

| Carga                                 | Modelos          | Datasets                          | GPU                                        |
| ------------------------------------- | ---------------- | --------------------------------- | ------------------------------------------ |
| GNN 1000 ép., dataset grande          | CPGRec           | SCGRec (2–3,9 M u)                | **A100 40 GB**                             |
| GNN 1000 ép., mediano                 | CPGRec           | Kozyriev (sub 500 K), UCSD (62 K) | **L4** (A100 si hay)                       |
| Deep CTR, catálogo chico              | FM/DeepFM/DeepNN | UCSD                              | **T4**                                     |
| Deep CTR, catálogo grande (full-rank) | FM/DeepFM/DeepNN | Kozyriev (22,6 K ítems)           | **L4 + High-RAM**                          |
| Encoding de contenido                 | CB-SBERT         | todos                             | **T4** (1 vez; el ranking es en CPU)       |
| Factorización implícita               | ALS, BPR         | UCSD, Kozyriev                    | **CPU High-RAM**                           |
| Factorización implícita               | ALS, BPR         | SCGRec                            | **High-RAM obligatorio** (subsample 2 M u) |
| Cheuque per-user (F3)                 | ALS + deep       | UCSD, Kozyriev                    | **T4** (deep) + CPU (ALS)                  |

---

## 8. Scripts

- **`recsys_protocol.py`** — módulo común importado por los cuadernillos: fija la
  semilla global (`set_global_seed`), filtro `iterative_k_core`, construcción de
  _splits_ y el arnés de métricas (Recall/NDCG/Hit/Precision + Coverage/entropía/
  _novelty_/ILD). Garantiza que todos los cuadernillos midan igual.
- **`generar_figuras.py`** — genera las figuras del informe a partir de las cifras
  consolidadas de los cuadernillos ejecutados (paleta Okabe-Ito, apta para daltonismo).
  Salida en `figs/`: efecto del protocolo, hallazgo del catálogo, _long-tail_,
  _trade-off_ accuracy/diversidad, distribución de actividad y _bump chart_ de ranking.
  Uso: `python generar_figuras.py`.
