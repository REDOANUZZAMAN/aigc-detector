"""Compare detectors and ensembles on bench/results/scores.csv.

    /root/projects/aigc-detect/.venv/bin/python bench/analyze.py > bench/results/report.md

Thresholds are tuned on the pilot set only and reported on the held-out `new`
set, so the ensemble can't be flattered by fitting the data it is judged on.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "results" / "scores.csv")))
DET = ["swinv2", "commfor", "siglip"]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cols(sel):
    y = np.array([int(r["label"]) for r in sel])
    s = {d: np.array([float(r[d]) for r in sel]) for d in DET}
    # Ensembles: mean log-odds (soft) and min probability (both must agree).
    s["swin+commfor mean"] = sigmoid((logit(s["swinv2"]) + logit(s["commfor"])) / 2)
    s["swin+commfor min"] = np.minimum(s["swinv2"], s["commfor"])
    s["all3 mean"] = sigmoid(sum(logit(s[d]) for d in DET) / 3)
    return y, s


def boot_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if 0 < y[b].sum() < len(b):
            vals.append(roc_auc_score(y[b], p[b]))
    return np.percentile(vals, [2.5, 97.5])


pilot = [r for r in rows if r["set"] == "pilot"]
new = [r for r in rows if r["set"] == "new"]
arms = sorted({r["arm"] for r in pilot})

print("# Second-detector benchmark\n")
print(f"{len(rows)} scored images: pilot {len(pilot)} (9 arms x 50), held-out new {len(new)} "
      f"({sum(r['arm'] == 'div2k' for r in new)} DIV2K real, "
      f"{sum(r['arm'] == 'flux' for r in new)} FLUX, {sum(r['arm'] == 'sdxl' for r in new)} SDXL).\n")

y, s = cols(new)
print("## AUC on held-out set (95% bootstrap CI)\n")
print("| detector | all new | real vs FLUX | real vs SDXL |\n|---|---|---|---|")
for d in s:
    parts = []
    for gen in (None, "flux", "sdxl"):
        sel = [r for r in new if r["arm"] == "div2k" or gen is None or r["arm"] == gen]
        yy, ss = cols(sel)
        lo, hi = boot_auc(yy, ss[d])
        parts.append(f"{roc_auc_score(yy, ss[d]):.3f} [{lo:.2f}-{hi:.2f}]")
    print(f"| {d} | " + " | ".join(parts) + " |")

print("\n## AUC per pilot degradation arm\n")
print("| detector | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
for d in s:
    vals = []
    for a in arms:
        yy, ss = cols([r for r in pilot if r["arm"] == a])
        vals.append(f"{roc_auc_score(yy, ss[d]):.3f}")
    print(f"| {d} | " + " | ".join(vals) + " |")

# Verdict-band behaviour. The product question isn't AUC, it's: at the
# "likely AI" cut-off, how many real photos get accused, and how many fakes caught?
print("\n## \"Likely AI\" band: false accusations vs catches\n")
print("Cut-off per detector = lowest value giving <=4% real FPR on the **pilot** (all arms); "
      "then applied unchanged to the held-out set.\n")
print("| detector | tuned cut-off | pilot FPR | pilot TPR | **held-out FPR** | **held-out TPR** "
      "| TPR FLUX | TPR SDXL |\n|---|---|---|---|---|---|---|---|")
yp, sp = cols(pilot)
for d in s:
    cands = np.unique(np.concatenate([sp[d], [0.99, 0.999, 0.9999]]))
    ok = [c for c in cands if (sp[d][yp == 0] >= c).mean() <= 0.04]
    c = min(ok)
    fpr_p, tpr_p = (sp[d][yp == 0] >= c).mean(), (sp[d][yp == 1] >= c).mean()
    fpr, tpr = (s[d][y == 0] >= c).mean(), (s[d][y == 1] >= c).mean()
    per = []
    for gen in ("flux", "sdxl"):
        m = np.array([r["arm"] == gen for r in new])
        per.append(f"{(s[d][m] >= c).mean():.2f}")
    print(f"| {d} | {c:.4f} | {fpr_p:.2f} | {tpr_p:.2f} | {fpr:.2f} | {tpr:.2f} | "
          + " | ".join(per) + " |")

print("\n## Current production bands (Swinv2 alone, 0.99 / 0.10) on held-out set\n")
sw = s["swinv2"]
for name, m in (("real", y == 0), ("fake", y == 1)):
    print(f"- {name}: likely AI {(sw[m] >= 0.99).mean():.2f} · inconclusive "
          f"{((sw[m] < 0.99) & (sw[m] > 0.10)).mean():.2f} · no strong signal {(sw[m] <= 0.10).mean():.2f}")

print("\n## Score correlation on held-out set (Spearman)\n")
from scipy.stats import spearmanr  # noqa: E402
for a, b in (("swinv2", "commfor"), ("swinv2", "siglip"), ("commfor", "siglip")):
    print(f"- {a} vs {b}: {spearmanr(s[a], s[b]).statistic:.2f}")


# ---------------------------------------------------------------------------
# Rigour checks: paired tests, a trivial baseline, CIs, per-arm operating points.


def proportion_confint(k, n, z=1.96, method="wilson"):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


PILOT_ARMS = Path("/root/projects/aigc-detect/corpus/arms")


def nbytes(r):
    if r["set"] == "pilot":
        return (PILOT_ARMS / r["arm"] / f"{r['image']}.jpg").stat().st_size
    return (HERE / "data" / r["arm"] / f"{r['image']}.jpg").stat().st_size


print("\n## Paired bootstrap: AUC gain over Swinv2 on held-out set\n")
print("Resamples images jointly, so both detectors see the same bootstrap sample.\n")
print("| detector | ΔAUC | 95% CI | P(Δ ≤ 0) |\n|---|---|---|---|")
rng = np.random.default_rng(1)
B = [rng.choice(len(y), len(y)) for _ in range(4000)]
B = [b for b in B if 0 < y[b].sum() < len(b)]
for d in s:
    if d == "swinv2":
        continue
    deltas = np.array([roc_auc_score(y[b], s[d][b]) - roc_auc_score(y[b], s["swinv2"][b]) for b in B])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"| {d} | {roc_auc_score(y, s[d]) - roc_auc_score(y, s['swinv2']):+.3f} | "
          f"[{lo:+.3f}, {hi:+.3f}] | {(deltas <= 0).mean():.3f} |")

print("\n## Trivial baseline: normalized file size alone\n")
print("Every image is JPEG Q95 at 1024², so size measures compressibility (content "
      "entropy), not provenance. Smaller file → predicted fake.\n")
size_new = -np.array([nbytes(r) for r in new], float)
lo, hi = boot_auc(y, size_new)
print(f"- held-out: AUC {roc_auc_score(y, size_new):.3f} [{lo:.2f}-{hi:.2f}] "
      f"(Swinv2 {roc_auc_score(y, s['swinv2']):.3f}, swin+commfor mean "
      f"{roc_auc_score(y, s['swin+commfor mean']):.3f})")
for gen in ("flux", "sdxl"):
    sel = [r for r in new if r["arm"] in ("div2k", gen)]
    yy = np.array([int(r["label"]) for r in sel])
    print(f"- real vs {gen}: AUC {roc_auc_score(yy, -np.array([nbytes(r) for r in sel], float)):.3f}")

print("\n## Operating points with 95% Wilson CIs (held-out set)\n")
print("| detector @ cut-off | FPR (reals flagged) | TPR (fakes caught) |\n|---|---|---|")
ops = {"swinv2": 0.99, "swin+commfor mean": None}
for d in ops:
    cands = np.unique(np.concatenate([sp[d], [0.99]]))
    c = ops[d] or min(x for x in cands if (sp[d][yp == 0] >= x).mean() <= 0.04)
    ops[d] = c
    fp, n0 = int((s[d][y == 0] >= c).sum()), int((y == 0).sum())
    tp, n1 = int((s[d][y == 1] >= c).sum()), int((y == 1).sum())
    f_lo, f_hi = proportion_confint(fp, n0, method="wilson")
    t_lo, t_hi = proportion_confint(tp, n1, method="wilson")
    print(f"| {d} @ {c:.4f} | {fp}/{n0} = {fp / n0:.3f} [{f_lo:.3f}-{f_hi:.3f}] | "
          f"{tp}/{n1} = {tp / n1:.3f} [{t_lo:.3f}-{t_hi:.3f}] |")

print("\n## Operating points per degradation arm (pilot, same cut-offs)\n")
print("| arm | Swinv2@0.99 FPR | Swinv2 TPR | ens FPR | ens TPR |\n|---|---|---|---|---|")
for a in arms:
    yy, ss = cols([r for r in pilot if r["arm"] == a])
    cells = []
    for d in ("swinv2", "swin+commfor mean"):
        cells += [f"{(ss[d][yy == 0] >= ops[d]).mean():.2f}", f"{(ss[d][yy == 1] >= ops[d]).mean():.2f}"]
    print(f"| {a} | " + " | ".join(cells) + " |")

print("\n## Score movement under degradation (mean |Δ| vs A0, pilot)\n")
print("A detector that reads generation artefacts should move when artefacts are destroyed.\n")
a0 = {r["image"]: r for r in pilot if r["arm"] == "A0"}
print("| detector | " + " | ".join(arms[1:]) + " |\n|---|" + "---|" * (len(arms) - 1))
for d in DET:
    vals = []
    for a in arms[1:]:
        diffs = [abs(float(r[d]) - float(a0[r["image"]][d])) for r in pilot if r["arm"] == a]
        vals.append(f"{np.mean(diffs):.3f}")
    print(f"| {d} | " + " | ".join(vals) + " |")
