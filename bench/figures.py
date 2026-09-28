"""Paper figures from bench/results/scores.csv -> paper/figs/*.pdf.

    /root/projects/aigc-detect/.venv/bin/python bench/figures.py

Colours are fixed per detector (validated categorical slots 1-4, light mode);
line style is a second cue so the figures survive greyscale printing.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": .5, "xtick.major.width": .5,
    "ytick.major.width": .5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": .02,
})
COL = 3.45  # IEEE column width, inches
GRID = dict(color="#e4e3df", linewidth=.4)
SERIES = {  # entity -> (label, colour, dash)
    "swinv2": ("Swinv2", "#2a78d6", "-"),
    "commfor": ("CommFor-384", "#eb6834", (0, (4, 1.5))),
    "siglip": ("SigLIP", "#1baf7a", (0, (1, 1.2))),
    "ens": ("Swinv2+CommFor", "#eda100", (0, (5, 1.5, 1, 1.5))),
}

rows = list(csv.DictReader(open(HERE / "results" / "scores.csv")))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def arrays(sel):
    y = np.array([int(r["label"]) for r in sel])
    s = {d: np.array([float(r[d]) for r in sel]) for d in ("swinv2", "commfor", "siglip")}
    s["ens"] = 1 / (1 + np.exp(-(logit(s["swinv2"]) + logit(s["commfor"])) / 2))
    return y, s


new = [r for r in rows if r["set"] == "new"]
pilot = [r for r in rows if r["set"] == "pilot"]

# ---- Fig. 1: ROC on the held-out set --------------------------------------
y, s = arrays(new)
fig, ax = plt.subplots(figsize=(COL, 2.55))
ax.plot([0, 1], [0, 1], color="#bdbcb6", lw=.6, ls=(0, (2, 2)))
for k, (label, c, ls) in SERIES.items():
    fpr, tpr, _ = roc_curve(y, s[k])
    ax.plot(fpr, tpr, color=c, ls=ls, lw=1.4,
            label=f"{label} (AUC {roc_auc_score(y, s[k]):.3f})")
ax.set(xlabel="False positive rate (real images flagged)",
       ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
ax.grid(**GRID)
ax.legend(loc="lower right", handlelength=2.6)
fig.savefig(OUT / "roc_heldout.pdf")

# ---- Fig. 2: AUC per degradation arm ---------------------------------------
arms = sorted({r["arm"] for r in pilot})
names = {"A0": "clean", "A1": "JPEG 90", "A2": "JPEG 75", "A3": "JPEG 60", "A4": "resize ×0.7",
         "A5": "resize ×0.5", "A6": "blur", "A7": "noise", "A8": "stacked"}
fig, ax = plt.subplots(figsize=(COL, 2.3))
x = np.arange(len(arms))
for k, (label, c, ls) in SERIES.items():
    auc = [roc_auc_score(*(lambda yy, ss: (yy, ss[k]))(*arrays([r for r in pilot if r["arm"] == a])))
           for a in arms]
    ax.plot(x, auc, color=c, ls=ls, lw=1.4, marker="o", ms=3.2, mec="white", mew=.6, label=label)
ax.set_xticks(x, [names[a] for a in arms], rotation=35, ha="right")
ax.set(ylabel="ROC AUC", ylim=(.6, 1.0))
ax.grid(axis="y", **GRID)
ax.legend(loc="lower left", ncol=2, handlelength=2.6)
fig.savefig(OUT / "auc_by_arm.pdf")

# ---- Fig. 3: score distributions, Swinv2 vs ensemble (held-out) ------------
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85), sharey=True)
bins = np.linspace(-14, 14, 29)
for ax, (k, cut, title) in zip(axes, (("swinv2", .99, "Swinv2"),
                                      ("ens", .6437, "Swinv2+CommFor"))):
    for lab, name, c in ((0, "real", "#2a78d6"), (1, "generated", "#eb6834")):
        ax.hist(logit(s[k][y == lab]), bins=bins, color=c, alpha=.55 if lab else .7,
                edgecolor="white", linewidth=.4, label=name)
    ax.axvline(logit(cut), color="#52514e", lw=.7, ls=(0, (3, 2)))
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("score (log-odds)")
    ax.grid(axis="y", **GRID)
axes[0].set_ylabel("images")
for ax, cut in zip(axes, (.99, .6437)):
    top = ax.get_ylim()[1]
    ax.text(logit(cut) + .6, top * .9, f"cut-off {cut:.2f}", fontsize=6, color="#52514e")
axes[0].legend(loc="upper left", handlelength=1.2)
fig.savefig(OUT / "score_dist.pdf")
print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
