# -*- coding: utf-8 -*-
"""
Genera las figuras del paper final (H3) — IIC3633 2026-1.

Todas las cifras provienen de los notebooks ejecutados del repositorio
(ver H3/RESULTADOS_*.md y H3/ESTADO_EXPERIMENTOS.md):
  - F2 (protocolo comun, full-ranking):
      CPGRec_comparativo_multiseed_{ucsd,kozyriev}_corregido_ejecutado.ipynb,
      CPGRec_comparativo_multiseed_T2_corregidoejecutado.ipynb,
      bpr_{ucsd,kozyriev,scgrec}_h3*_ejecutado.ipynb,
      sbert_{ucsd,kozyriev}_h3 ejecutado.ipynb,
      deep_kozyriev_h3_ejecutado.ipynb
  - F3 (protocolo per-user, reproduccion Cheuque'19):
      cheuque_ucsd_inicial.ipynb, cheuque_kozyriev_ejecutado.ipynb

Salida: figs/fig_protocolo.pdf, figs/fig_catalogo.pdf, figs/fig_longtail.pdf
Uso:    python generar_figuras.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

# Paleta consistente entre figuras (Okabe-Ito, apta para daltonismo)
C = {
    "MostPop":  "#999999",
    "ALS":      "#0072B2",
    "BPR":      "#56B4E9",
    "CB-SBERT": "#009E73",
    "CPGRec":   "#D55E00",
    "FM":       "#E69F00",
    "DeepFM":   "#CC79A7",
    "DeepNN":   "#F0E442",
}

# ---------------------------------------------------------------------------
# Figura 1 — Efecto del protocolo de evaluacion (NDCG@10 de FM/DeepFM/DeepNN)
#   per-user con playtime (F3) -> per-user sin playtime (F3) -> full-ranking (F2)
# ---------------------------------------------------------------------------
protocol = {
    "UCSD (tasa base 24,7 %)": {
        "FM":     [0.992, 0.708, 0.1375],
        "DeepFM": [0.992, 0.715, 0.1408],
        "DeepNN": [0.992, 0.731, 0.0768],
    },
    "Kozyriev (tasa base 68,5 %)": {
        "FM":     [0.9985, 0.9051, 0.0232],
        "DeepFM": [0.9976, 0.9090, 0.0255],
        "DeepNN": [0.9985, 0.9249, 0.0096],
    },
}
etapas = ["per-user\ncon playtime", "per-user\nsin playtime", "full-ranking\n(protocolo común)"]

def fmt(v, dec=2):
    return f"{v:.{dec}f}".replace(".", ",")

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), sharey=True)
ancho = 0.24
for ax, (ds, modelos) in zip(axes, protocol.items()):
    for j, (m, ys) in enumerate(modelos.items()):
        xs = [i + (j - 1) * ancho for i in range(3)]
        ax.bar(xs, ys, width=ancho, color=C[m], label=m, edgecolor="white", lw=0.4)
        for x, v in zip(xs, ys):
            ax.annotate(fmt(v, 2 if v > 0.2 else 3), (x, v), ha="center", va="bottom",
                        fontsize=6.2, rotation=90, xytext=(0, 1.5),
                        textcoords="offset points")
    ax.set_xticks(range(3))
    ax.set_xticklabels(etapas)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.18)
    ax.set_title(ds)
    ax.axvspan(-0.55, 1.5, color="#D55E00", alpha=0.05, lw=0)
    ax.axvspan(1.5, 2.55, color="#009E73", alpha=0.07, lw=0)
axes[0].set_ylabel("NDCG@10")
axes[0].legend(loc="center right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_protocolo.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura 2 — Hallazgo del catalogo: Recall@5 (F2) por modelo x dataset
#   (escala propia por panel: los absolutos no son comparables entre datasets)
# ---------------------------------------------------------------------------
f2 = {
    "SCGRec\n2.675 juegos (denso)": [
        ("MostPop", 0.2128), ("ALS", 0.3752), ("BPR", 0.4145),
        ("CPGRec", 0.4080),
    ],
    "UCSD\n9.198 juegos (medio)": [
        ("MostPop", 0.1191), ("ALS", 0.1050), ("BPR", 0.1405),
        ("CB-SBERT", 0.0128), ("CPGRec", 0.1440),
        ("FM", 0.0905), ("DeepFM", 0.0919), ("DeepNN", 0.0402),
    ],
    "Kozyriev\n22.590 juegos (disperso)": [
        ("MostPop", 0.0289), ("ALS", 0.0586), ("BPR", 0.0392),
        ("CB-SBERT", 0.0159), ("CPGRec", 0.0440),
        ("FM", 0.0239), ("DeepFM", 0.0263), ("DeepNN", 0.0092),
    ],
}

fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.15))
for ax, (ds, filas) in zip(axes, f2.items()):
    nombres = [n for n, _ in filas]
    vals = [v for _, v in filas]
    colores = [C[n] for n in nombres]
    best = max(vals)
    bars = ax.bar(range(len(vals)), vals, color=colores,
                  edgecolor=["black" if v == best else "none" for v in vals],
                  linewidth=0.9)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}".replace(".", ","),
                    (i, v), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=55, ha="right", fontsize=6.8)
    ax.set_title(ds, fontsize=8)
    ax.set_ylim(0, best * 1.22)
axes[0].set_ylabel("Recall@5")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_catalogo.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura 3 — Long-tail: NDCG@10 por nivel de actividad del usuario
# ---------------------------------------------------------------------------
buckets = ["2–5", "6–20", "21–50", "51+"]
longtail = {
    "SCGRec (denso)": {
        "MostPop": [0.2739, 0.2196, 0.1566, 0.1330],
        "ALS":     [0.5933, 0.4000, 0.1751, 0.0996],
        "BPR":     [0.6271, 0.4309, 0.2281, 0.1795],
        "CPGRec":  [0.5977, 0.4303, 0.2514, 0.2100],
    },
    "UCSD (medio)": {
        "MostPop": [0.2235, 0.1768, 0.1561, 0.1628],
        "ALS":     [0.1999, 0.1900, 0.1541, 0.1110],
        "BPR":     [0.3249, 0.2511, 0.1769, 0.1539],
        "CPGRec":  [0.2208, 0.2133, 0.1967, 0.2051],
    },
    "Kozyriev (disperso)": {
        "MostPop": [0.0292, 0.0277, 0.0258, 0.0244],
        "ALS":     [0.0534, 0.0585, 0.0551, 0.0534],
        "BPR":     [0.0370, 0.0380, 0.0362, 0.0446],
        "CPGRec":  [0.0391, 0.0446, 0.0494, 0.0537],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.0))
for ax, (ds, modelos) in zip(axes, longtail.items()):
    for m, ys in modelos.items():
        ax.plot(range(4), ys, marker="o", ms=3.5, lw=1.4, color=C[m], label=m)
    ax.set_xticks(range(4))
    ax.set_xticklabels(buckets)
    ax.set_title(ds)
    ax.set_xlabel("interacciones en train")
axes[0].set_ylabel("NDCG@10")
axes[2].legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_longtail.pdf"), bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# Figuras del ANEXO
# ===========================================================================

# ---------------------------------------------------------------------------
# Figura E1 — Trade-off accuracy (R@5) vs diversidad (Cov_total@10), por dataset
# ---------------------------------------------------------------------------
tradeoff = {
    "SCGRec (denso)": {
        "MostPop": (0.2128, 4.14), "ALS": (0.3752, 19.97),
        "BPR": (0.4145, 14.51), "CPGRec": (0.4080, 13.99),
    },
    "UCSD (medio)": {
        "MostPop": (0.1191, 17.50), "ALS": (0.1050, 21.96),
        "BPR": (0.1405, 20.10), "CB-SBERT": (0.0128, 25.64),
        "CPGRec": (0.1440, 19.50), "FM": (0.0905, 18.33),
        "DeepFM": (0.0919, 17.66), "DeepNN": (0.0402, 16.78),
    },
    "Kozyriev (disperso)": {
        "MostPop": (0.0289, 29.24), "ALS": (0.0586, 31.91),
        "BPR": (0.0392, 33.02), "CB-SBERT": (0.0159, 31.72),
        "CPGRec": (0.0440, 31.56), "FM": (0.0239, 28.20),
        "DeepFM": (0.0263, 27.90), "DeepNN": (0.0092, 32.90),
    },
}

# desplazamientos de etiqueta para puntos que se solapan (ha, dx, dy)
offs = {
    ("SCGRec (denso)", "CPGRec"): ("left", 4, -9),
    ("UCSD (medio)", "FM"): ("right", -4, 4),
    ("UCSD (medio)", "DeepFM"): ("left", 4, -9),
    ("Kozyriev (disperso)", "FM"): ("right", -4, 4),
    ("Kozyriev (disperso)", "DeepFM"): ("left", 4, -9),
    ("Kozyriev (disperso)", "ALS"): ("right", -2, 6),
    ("Kozyriev (disperso)", "CPGRec"): ("left", 4, -9),
}

fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2))
for ax, (ds, pts) in zip(axes, tradeoff.items()):
    for m, (x, y) in pts.items():
        ax.scatter(x, y, s=42, color=C[m], edgecolor="black", lw=0.4, zorder=3)
        ha, dx, dy = offs.get((ds, m), ("left", 4, 3))
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=6, zorder=4)
    ax.set_title(ds)
    ax.set_xlabel("Recall@5 (accuracy)")
    xs = [v[0] for v in pts.values()]
    ax.set_xlim(min(xs) - 0.15 * (max(xs) - min(xs)), max(xs) * 1.25)
axes[0].set_ylabel("Cov_total@10 (diversidad)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tradeoff.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura E2 — Distribución de la población evaluada por nivel de actividad
# ---------------------------------------------------------------------------
poblacion = {  # n usuarios evaluados por bucket [2-5, 6-20, 21-50, 51+]
    "UCSD":     [1750, 12122, 18396, 25122],
    "Kozyriev": [88748, 93068, 15310, 2874],
    "SCGRec":   [36313, 100298, 41012, 22377],
}
bucket_cols = ["#c6dbef", "#6baed6", "#2171b5", "#0b2a45"]
bucket_lbls = ["2–5 (cold)", "6–20", "21–50", "51+ (heavy)"]

fig, ax = plt.subplots(figsize=(5.4, 1.9))
ys = range(len(poblacion))
for j in range(4):
    left = [sum(v[:j]) / sum(v) * 100 for v in poblacion.values()]
    vals = [v[j] / sum(v) * 100 for v in poblacion.values()]
    ax.barh(list(ys), vals, left=left, color=bucket_cols[j],
            label=bucket_lbls[j], height=0.62)
    for y, (l, v) in enumerate(zip(left, vals)):
        if v > 6:
            ax.annotate(f"{v:.0f} %", (l + v / 2, y), ha="center", va="center",
                        fontsize=6.5, color="white" if j >= 2 else "#1b2838")
ax.set_yticks(list(ys))
ax.set_yticklabels(list(poblacion.keys()))
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% de los usuarios evaluados")
ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_actividad.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura E3 — Bump chart: ranking por R@5 al crecer el catalogo
# ---------------------------------------------------------------------------
orden_ds = ["SCGRec\n2.675 j", "UCSD\n9.198 j", "Kozyriev\n22.590 j"]
rankings = {  # posicion (1 = mejor) por dataset, R@5 entre {MostPop, ALS, BPR, CPGRec}
    "BPR":     [1, 2, 3],
    "CPGRec":  [2, 1, 2],
    "ALS":     [3, 4, 1],
    "MostPop": [4, 3, 4],
}

fig, ax = plt.subplots(figsize=(4.6, 2.2))
for m, ys in rankings.items():
    ax.plot([0, 1, 2], ys, marker="o", ms=6, lw=2, color=C[m], zorder=3)
    ax.annotate(m, (0, ys[0]), textcoords="offset points", xytext=(-8, 0),
                ha="right", va="center", fontsize=7.5, color=C[m], fontweight="bold")
    ax.annotate(m, (2, ys[2]), textcoords="offset points", xytext=(8, 0),
                ha="left", va="center", fontsize=7.5, color=C[m], fontweight="bold")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(orden_ds)
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(["1.º", "2.º", "3.º", "4.º"])
ax.invert_yaxis()
ax.set_ylabel("Ranking por Recall@5")
ax.set_xlim(-0.75, 2.75)
ax.spines["left"].set_visible(False)
ax.grid(axis="y", lw=0.4, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_bump.pdf"), bbox_inches="tight")
plt.close(fig)

print("Figuras escritas en", OUT)
