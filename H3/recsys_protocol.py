"""
recsys_protocol.py — Protocolo experimental común (Proyecto RecSys H3)
======================================================================

Módulo único e importable que estandariza el **rigor experimental** (§4.4 del
`PLAN_H3.md`) para TODOS los datasets y notebooks. Reemplaza el código de
seed/muestreo/split/métricas que hoy se redefine (y diverge) en cada notebook.

Por qué existe (hallazgos de la auditoría §4.4):
  - #2  El 10% de usuarios muestreado NO era el mismo entre notebooks: BPR
        ordenaba por fecha antes del `.sample()` y SBERT/CPGRec por orden de
        archivo, así que con la misma seed `Series.sample` elegía POSICIONES
        iguales pero `user_id` distintos → eval pools distintos → tablas
        comparativas no estrictamente comparables. Aquí `sample_users()`
        muestrea SIEMPRE sobre un universo ordenado (`np.sort`), por lo que el
        subconjunto es idéntico en todos los notebooks dada la misma seed.
  - #3  La seed del GNN (3407/2023) estaba desacoplada del harness (42).
        Aquí hay UNA seed de proyecto (`SEED`) y `set_global_seed()` cubre
        random/numpy/torch/cuda + cudnn determinista.

Diseño:
  - Genérico en nombres de columna (`ProtocolConfig`) → sirve igual a Kozyriev
    (`is_recommended`/`hours`/`date`), UCSD (`recommend`/`playtime_forever`) y
    SCGRec. `pos_col=None` ⇒ todas las interacciones son positivas.
  - El split leave-one-out temporal de aquí es para NUESTRA evaluación
    cross-dataset. SCGRec/CPGRec traen split OFICIAL: para *reproducir* esos
    papers se usa su split, NO `build_splits()`. Las métricas (sección 4) sí se
    reutilizan en ambos protocolos.

Uso típico en un notebook (tras clonar el repo y tenerlo en sys.path):

    from recsys_protocol import (
        SEED, ProtocolConfig, set_global_seed, iterative_k_core,
        build_splits, positives, popularity_counts,
        evaluate_at_ks, coverage_at_k, novelty_at_k, ild_at_k,
        category_diversity, hit_ratio_at_k, reproducibility_note,
    )

    set_global_seed()                       # una vez, al inicio
    recs = iterative_k_core(recs, 5, 20)    # mismo k-core en todos
    cfg  = ProtocolConfig(frac_eval=0.10, frac_train=0.10)
    sp   = build_splits(recs, cfg)          # eval pool determinista + LOO
    # sp.train_df / sp.test_item_per_user / sp.train_items_per_user / sp.eval_users
"""
from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0. Seed única de proyecto
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42  # ÚNICA seed del proyecto. No redefinir por notebook.

# Versión del protocolo. Se imprime en set_global_seed para verificar de un
# vistazo QUÉ versión del módulo cargó cada corrida (clave en Colab, donde el
# módulo se trae por git clone/pull). Subir al cambiar la lógica del split/métricas.
PROTOCOL_VERSION = "2026-06-22c (split: desempate determinista [user,date,item])"


def set_global_seed(seed: int = SEED, deterministic_torch: bool = True,
                    verbose: bool = True) -> None:
    """Fija la seed global en random/numpy/torch/cuda + cudnn determinista.

    Llamar UNA vez al inicio de cada notebook. Cubre el muestreo/split de este
    módulo y el entrenamiento del GNN. No cubre la RNG interna de `implicit`
    (ALS/BPR): a esos modelos hay que pasarles `random_state=SEED` explícito.

    Nota: `PYTHONHASHSEED` solo surte efecto pleno si se fija ANTES de lanzar el
    intérprete; se setea aquí por completitud.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    backend = "numpy/random"
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        backend += f"/torch (cuda={torch.cuda.is_available()}, determinista={deterministic_torch})"
    except ImportError:
        backend += " (torch no instalado)"
    if verbose:
        print(f"[protocol] v{PROTOCOL_VERSION}")
        print(f"[protocol] seed global = {seed} | {backend}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuración del protocolo
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProtocolConfig:
    """Parámetros del protocolo. Toda diferencia entre datasets debe ser
    deliberada y quedar registrada cambiando estos campos (no a mano)."""
    seed: int = SEED
    # Muestreo de usuarios
    frac_eval: float = 0.10        # universo de EVALUACIÓN (fijo, define a quién se mide)
    frac_train: float = 0.10       # universo de ENTRENAMIENTO (palanca; >= frac_eval; 1.0 = todos)
    max_eval_users: int = 2000     # tope de usuarios evaluados (None = sin tope)
    # k-core
    min_user: int = 5
    min_game: int = 20
    # Métricas
    ks: Sequence[int] = (5, 10, 20)
    # Nombres de columna (cambian por dataset)
    user_col: str = "user_id"
    item_col: str = "app_id"
    time_col: str = "date"
    pos_col: Optional[str] = "is_recommended"  # None ⇒ toda interacción es positiva

    def as_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Muestreo determinista (FIX hallazgo #2) y k-core
# ─────────────────────────────────────────────────────────────────────────────
def _unique_sorted_users(data, user_col: str = "user_id") -> np.ndarray:
    """Universo de usuarios ÚNICO y ORDENADO. Ordenar es lo que hace que el
    muestreo sea independiente del orden de filas del DataFrame de origen."""
    if isinstance(data, pd.DataFrame):
        vals = data[user_col].to_numpy()
    elif isinstance(data, pd.Series):
        vals = data.to_numpy()
    else:
        vals = np.asarray(data)
    return np.sort(pd.unique(vals))


def sample_users(data, frac: float, seed: int = SEED,
                 *, user_col: str = "user_id") -> np.ndarray:
    """Muestrea una fracción de usuarios de forma REPRODUCIBLE y ORDEN-INDEPENDIENTE.

    A diferencia de `Series.sample(random_state=seed)`, el resultado NO depende
    del orden en que vienen las filas: muestreamos sobre `np.sort(unique)`, así
    que dos notebooks con el mismo dataset y seed obtienen EXACTAMENTE el mismo
    subconjunto. `frac >= 1.0` devuelve todos los usuarios.

    El tamaño usa `round(frac * n)` para coincidir con el conteo de
    `pandas.sample(frac=...)` (p.ej. 0.10 × 1.905.447 = 190.545).
    """
    users = _unique_sorted_users(data, user_col)
    if frac >= 1.0:
        return users
    n = int(round(frac * len(users)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(users), size=n, replace=False)
    return np.sort(users[idx])


def iterative_k_core(df: pd.DataFrame, min_user: int, min_game: int,
                     *, user_col: str = "user_id", item_col: str = "app_id",
                     max_iter: int = 30, verbose: bool = False) -> pd.DataFrame:
    """k-core iterativo (≥min_user por usuario, ≥min_game por ítem) hasta punto fijo.
    Idéntico al de H1/H2; centralizado para que sea el mismo en todos lados."""
    for it in range(1, max_iter + 1):
        n0 = len(df)
        uc = df[user_col].value_counts()
        gc = df[item_col].value_counts()
        df = df[df[user_col].isin(uc[uc >= min_user].index)
                & df[item_col].isin(gc[gc >= min_game].index)].copy()
        if verbose:
            print(f"  iter {it}: {n0:,} → {len(df):,}")
        if len(df) == n0:
            break
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Split leave-one-out temporal + política eval/train pool
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Splits:
    """Salida de `build_splits`. Todo lo que un notebook necesita aguas abajo."""
    train_df: pd.DataFrame
    test_pos: pd.DataFrame
    test_item_per_user: dict          # user_id -> ítem retenido (último positivo)
    train_items_per_user: dict        # user_id -> set de ítems en train
    eval_users: list                  # usuarios efectivamente evaluados (≤ max_eval_users)
    eval_pool: np.ndarray             # universo de evaluación (frac_eval)
    train_pool: Optional[np.ndarray]  # universo de entrenamiento (None = todos)
    n_train_users: int
    n_items_train: int
    config: ProtocolConfig

    def summary(self) -> str:
        return (f"train_users={self.n_train_users:,} | items_train={self.n_items_train:,} | "
                f"eval_pool={len(self.eval_pool):,} | eval_users={len(self.eval_users):,}")


def build_splits(df: pd.DataFrame, cfg: ProtocolConfig = ProtocolConfig(),
                 *, verbose: bool = True) -> Splits:
    """Construye el split leave-one-out temporal con la política de pools.

    Política (la de `H2_cpgrec`, generalizada y con muestreo arreglado):
      - `eval_pool`  = `frac_eval` de usuarios (FIJO) → define a quién se evalúa,
        comparable entre corridas/notebooks/modelos.
      - `train_pool` = `frac_train` de usuarios, SIEMPRE ⊇ `eval_pool`. Los
        usuarios "solo-train" aportan TODO su historial (no se les quita ítem).
        `frac_train == frac_eval` reproduce el régimen clásico de H2; subirlo
        (hasta 1.0) añade señal sin cambiar el set de evaluación.
      - Leave-one-out SOLO sobre `eval_pool`: el último ítem por fecha va a test;
        se evalúan solo los usuarios cuyo ítem de test es positivo (`pos_col`).

    Asume que `df` ya pasó por `iterative_k_core`. Verifica ausencia de fuga:
    el ítem retenido se quita de train y nunca queda en `train_items_per_user`.
    """
    uc, ic, tc, pc = cfg.user_col, cfg.item_col, cfg.time_col, cfg.pos_col

    eval_pool = sample_users(df, cfg.frac_eval, cfg.seed, user_col=uc)
    eval_set = set(eval_pool.tolist())

    if cfg.frac_train >= 1.0:
        recs_train = df
        train_pool = None
    else:
        tp = set(sample_users(df, cfg.frac_train, cfg.seed, user_col=uc).tolist()) | eval_set
        recs_train = df[df[uc].isin(tp)].copy()
        train_pool = np.sort(np.fromiter(tp, dtype=eval_pool.dtype))

    # El tiempo puede venir como string (algunos notebooks difieren el parseo por
    # velocidad): parsear SOLO el subconjunto de train_pool y luego ordenar. Las
    # fechas ISO (YYYY-MM-DD) ordenan bien como string, pero parseamos por robustez.
    if not pd.api.types.is_datetime64_any_dtype(recs_train[tc]):
        recs_train = recs_train.copy()
        recs_train[tc] = pd.to_datetime(recs_train[tc], errors="coerce")
    # DESEMPATE DETERMINISTA: las fechas suelen ser a nivel de día → muchos usuarios
    # tienen varias interacciones en su último día. Ordenar también por `ic` (ítem)
    # hace que el ítem retenido (tail(1)) sea el MISMO sin importar el orden de filas
    # de entrada → split idéntico entre notebooks (BPR/SBERT/CPGRec). Sin esto, el
    # empate se rompía según el orden de carga/dedup y cada notebook evaluaba otro set.
    recs_train = recs_train.sort_values([uc, tc, ic])
    is_eval = recs_train[uc].isin(eval_set)
    last_idx = recs_train[is_eval].groupby(uc).tail(1).index

    train_df = recs_train.drop(index=last_idx).reset_index(drop=True)
    test_df = recs_train.loc[last_idx].reset_index(drop=True)
    test_pos = test_df if pc is None else test_df[test_df[pc]].copy()

    train_items_per_user = train_df.groupby(uc)[ic].apply(set).to_dict()
    test_item_per_user = dict(zip(test_pos[uc], test_pos[ic]))
    # Solo evaluable quien tiene historial en train (excluye cold-start de eval)
    eval_users = [u for u in test_pos[uc].unique() if u in train_items_per_user]

    rng = np.random.default_rng(cfg.seed)
    if cfg.max_eval_users is not None and len(eval_users) > cfg.max_eval_users:
        eval_users = rng.choice(np.array(eval_users), size=cfg.max_eval_users,
                                replace=False).tolist()

    sp = Splits(
        train_df=train_df, test_pos=test_pos,
        test_item_per_user=test_item_per_user,
        train_items_per_user=train_items_per_user,
        eval_users=eval_users, eval_pool=eval_pool, train_pool=train_pool,
        n_train_users=int(train_df[uc].nunique()),
        n_items_train=int(train_df[ic].nunique()),
        config=cfg,
    )
    # Chequeo de fuga (barato): ningún ítem de test está en el historial de train del usuario
    leaks = sum(1 for u, it in test_item_per_user.items()
                if it in train_items_per_user.get(u, ()))
    if leaks:
        raise AssertionError(f"FUGA: {leaks} ítems de test presentes en train del mismo usuario")
    if verbose:
        print(f"[protocol] {sp.summary()} | fuga=0 ✓")
    return sp


def positives(df: pd.DataFrame, cfg: ProtocolConfig) -> pd.DataFrame:
    """Subconjunto positivo de interacciones (para matrices ALS/BPR, popularidad).
    Si `pos_col` es None, todas las interacciones cuentan como positivas."""
    return df if cfg.pos_col is None else df[df[cfg.pos_col]]


def popularity_counts(train_df: pd.DataFrame, cfg: ProtocolConfig) -> dict:
    """Conteo de interacciones positivas por ítem en train (para Novelty/fallback)."""
    return positives(train_df, cfg).groupby(cfg.item_col).size().to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Harness de métricas (idéntico a H2 para que los números coincidan)
# ─────────────────────────────────────────────────────────────────────────────
def precision_at_k(rec, rel, k) -> float:
    return sum(1 for x in rec[:k] if x in rel) / k


def recall_at_k(rec, rel, k) -> float:
    return (sum(1 for x in rec[:k] if x in rel) / len(rel)) if rel else 0.0


def ndcg_at_k(rec, rel, k) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(rec[:k]) if x in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_ratio_at_k(recs: dict, test_item: dict, k: int) -> float:
    """Hit Ratio@k (para comparación externa con SCGRec/CPGRec/DRGame).
    Con un único ítem retenido por usuario (LOO) equivale a Recall@k, pero se
    expone aparte porque esos papers reportan HR explícito."""
    vals = [1.0 if test_item[u] in set(rec[:k]) else 0.0
            for u, rec in recs.items() if u in test_item]
    return float(np.mean(vals)) if vals else 0.0


def evaluate_at_ks(recs: dict, test_item: dict, ks: Sequence[int] = (5, 10, 20)) -> dict:
    """Precision/Recall/NDCG @k sobre el ítem único retenido (full-ranking)."""
    acc = {f"{m}@{k}": [] for k in ks for m in ("Precision", "Recall", "NDCG")}
    n = 0
    for u, rec in recs.items():
        if u not in test_item:
            continue
        rel = {test_item[u]}
        n += 1
        for k in ks:
            acc[f"Precision@{k}"].append(precision_at_k(rec, rel, k))
            acc[f"Recall@{k}"].append(recall_at_k(rec, rel, k))
            acc[f"NDCG@{k}"].append(ndcg_at_k(rec, rel, k))
    out = {key: (float(np.mean(v)) if v else 0.0) for key, v in acc.items()}
    out["n_users"] = n
    return out


def coverage_at_k(recs: dict, catalog_size: int, k: int = 10) -> float:
    """Cobertura de ÍTEMS: fracción del catálogo que aparece en algún top-k."""
    return len({x for items in recs.values() for x in items[:k]}) / max(catalog_size, 1)


def novelty_at_k(recs: dict, popularity: dict, total: int, k: int = 10,
                 skip_unseen: bool = True) -> float:
    """Novedad = self-information media -log2(p(i)), con p(i)=popularity[i]/total.

    CONVENCIÓN DEL PROYECTO (estandarizada): `total = sum(popularity.values())`
    (nº total de interacciones positivas en train) y `skip_unseen=True` (ignora
    ítems con popularidad 0, es decir nunca vistos como positivos en train). Es
    la convención de H2 SBERT/CPGRec; H2 BPR usaba `total=nº usuarios` y default
    1 → distinta ESCALA. El denominador es un corrimiento constante en log, así
    que lo único imprescindible es usar la MISMA convención en todos los modelos
    de una misma tabla. `skip_unseen=False` cuenta los no vistos como si
    aparecieran 1 vez (replica el comportamiento viejo de BPR)."""
    scores = []
    for items in recs.values():
        nov = []
        for i in items[:k]:
            p = popularity.get(i, 0)
            if p > 0:
                nov.append(-np.log2(p / total))
            elif not skip_unseen:
                nov.append(-np.log2(1.0 / total))
        if nov:
            scores.append(float(np.mean(nov)))
    return float(np.mean(scores)) if scores else 0.0


def diversity_metrics(recs: dict, *, popularity: dict, total: int,
                      item_to_idx: dict, factors, n_catalog: int,
                      k: int = 10, assume_normalized: bool = False) -> dict:
    """Bundle Coverage/ILD/Novelty@k (espejo del `diversity_metrics` de H2 SBERT).
    Conveniencia para no repetir las tres llamadas en cada notebook; usa la
    convención canónica de Novelty (ver `novelty_at_k`)."""
    return {
        f"Coverage@{k}": coverage_at_k(recs, n_catalog, k),
        f"ILD@{k}": ild_at_k(recs, item_to_idx, factors, k, assume_normalized),
        f"Novelty@{k}": novelty_at_k(recs, popularity, total, k),
    }


def ild_at_k(recs: dict, item_to_idx: dict, factors, k: int = 10,
             assume_normalized: bool = False) -> float:
    """Intra-List Diversity = 1 - similitud coseno media entre pares del top-k.
    `factors` = embeddings/latentes del ítem (ndarray indexable por item_to_idx).
    Por defecto normaliza (evita el sesgo si los vectores no son unitarios)."""
    f = np.asarray(factors, dtype="float32")
    if not assume_normalized:
        norms = np.linalg.norm(f, axis=1, keepdims=True)
        f = f / np.where(norms == 0, 1.0, norms)
    scores = []
    for items in recs.values():
        idxs = [item_to_idx[i] for i in items[:k] if i in item_to_idx]
        if len(idxs) < 2:
            continue
        V = f[idxs]
        S = V @ V.T
        iu = np.triu_indices(len(idxs), k=1)
        scores.append(1.0 - float(S[iu].mean()))
    return float(np.mean(scores)) if scores else 0.0


def category_universe(cat_map: dict, item_universe: Sequence) -> set:
    """Conjunto de categorías alcanzables = unión de categorías sobre el catálogo."""
    u = set()
    for a in item_universe:
        u |= set(cat_map.get(a, ()))
    return u


def category_diversity(recs: dict, cat_map: dict, k: int = 10,
                       universe: Optional[set] = None,
                       item_universe: Optional[Sequence] = None) -> dict:
    """Diversidad de CATEGORÍAS (lo que CPGRec optimiza): GenreCov + GenreEnt.
      - CatCov  = nº de categorías distintas recomendadas / |universo de categorías|
      - Entropy = -Σ p·log2(p) sobre el reparto de categorías en los top-k
    `cat_map`: item_id -> iterable de categorías (género/dev/publisher)."""
    if universe is None:
        universe = category_universe(cat_map, item_universe or [])
    cnt = Counter()
    for rec in recs.values():
        for i in rec[:k]:
            for c in cat_map.get(i, ()):
                cnt[c] += 1
    tot = sum(cnt.values())
    if tot == 0:
        return {"CatCov": 0.0, "Entropy": 0.0}
    p = np.array(list(cnt.values()), dtype=float) / tot
    return {"CatCov": len(cnt) / max(len(universe), 1),
            "Entropy": float(-(p * np.log2(p)).sum())}


def activity_ndcg(recs: dict, test_item: dict, train_items: dict, k: int = 10,
                  bins=((2, 5, "2-5"), (6, 10, "6-10"),
                        (11, 20, "11-20"), (21, 9999, "21+"))) -> dict:
    """NDCG@k desagregado por nivel de actividad del usuario (largo de historial
    en train). Para el análisis de cola larga prometido en H2."""
    result = {}
    for lo, hi, label in bins:
        vals = [ndcg_at_k(rec, {test_item[u]}, k)
                for u, rec in recs.items()
                if u in test_item and lo <= len(train_items.get(u, ())) <= hi]
        result[label] = (float(np.mean(vals)) if vals else 0.0, len(vals))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Nota de reproducibilidad (para el paper)
# ─────────────────────────────────────────────────────────────────────────────
def reproducibility_note(cfg: ProtocolConfig, dataset_name: str = "") -> str:
    """Texto listo para la sección de reproducibilidad del paper/README."""
    head = f"Protocolo {dataset_name}".strip()
    return (
        f"{head}\n"
        f"  seed global         : {cfg.seed} (numpy/random/torch/cuda, cudnn determinista)\n"
        f"  muestreo usuarios   : eval={cfg.frac_eval:.2f} (fijo) · train={cfg.frac_train:.2f} "
        f"(⊇ eval) · determinista sobre np.sort(usuarios)\n"
        f"  k-core              : ≥{cfg.min_user} interac/usuario, ≥{cfg.min_game} interac/ítem (iterativo)\n"
        f"  split               : leave-one-out temporal (último ítem positivo a test)\n"
        f"  evaluación          : full-ranking sobre el catálogo · ≤{cfg.max_eval_users} usuarios · "
        f"K∈{tuple(cfg.ks)}\n"
        f"  columnas            : user={cfg.user_col} item={cfg.item_col} "
        f"time={cfg.time_col} pos={cfg.pos_col}\n"
        f"  ADVERTENCIA         : SCGRec/CPGRec usan su split OFICIAL para *reproducir*; "
        f"este leave-one-out es solo para evaluación cross-dataset (no mezclar)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Auto-test (no requiere los datasets reales)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    try:  # consola Windows (cp1252) ⇒ forzar UTF-8 para los símbolos del reporte
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Demuestra el FIX #2: el muestreo es independiente del orden de las filas.
    rng = np.random.default_rng(0)
    users = np.arange(1000)
    inter = np.repeat(users, 3)

    df_a = pd.DataFrame({"user_id": rng.permutation(inter)})          # un orden
    df_b = pd.DataFrame({"user_id": np.sort(inter)})                  # otro orden
    sa = sample_users(df_a, 0.10, SEED)
    sb = sample_users(df_b, 0.10, SEED)
    assert np.array_equal(sa, sb), "sample_users NO es orden-independiente"
    assert len(sa) == 100, f"tamaño esperado 100, obtenido {len(sa)}"

    # Contraste: el .sample() de pandas SÍ depende del orden (lo que rompía H2).
    old_a = set(pd.Series(df_a["user_id"].unique()).sample(frac=0.10, random_state=SEED))
    old_b = set(pd.Series(df_b["user_id"].unique()).sample(frac=0.10, random_state=SEED))
    print(f"sample_users (nuevo): A==B → {np.array_equal(sa, sb)} ✓  (n={len(sa)})")
    print(f"pandas.sample (viejo): A==B → {old_a == old_b}  "
          f"(solapamiento {len(old_a & old_b)}/{len(old_a)})")

    # Sanity de métricas con un caso trivial.
    recs = {1: [10, 20, 30], 2: [40, 50, 60]}
    test = {1: 20, 2: 99}  # user1 acierta en pos 2, user2 no acierta
    m = evaluate_at_ks(recs, test, ks=(1, 3))
    assert m["n_users"] == 2
    assert abs(m["Recall@3"] - 0.5) < 1e-9, m
    assert abs(m["Recall@1"] - 0.0) < 1e-9, m
    print(f"métricas sanity: NDCG@3={m['NDCG@3']:.4f} Recall@3={m['Recall@3']:.2f} ✓")
    print("\nTodos los auto-tests OK.")
