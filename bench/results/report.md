# Second-detector benchmark: report

Run 2026-09-28. Question: does adding a second detector to the production Swinv2
make the product's verdicts better, and if so, which detector and how should the
scores be combined? Full generated tables are in [`report-tables.md`](report-tables.md);
per-image scores are in [`scores.csv`](scores.csv); figures are in `paper/figs/`.

**Recommendation: ship Swinv2 + CommFor-384, averaged in log-odds space, with a
"likely AI" cut-off of 0.64.** On held-out images it raised AUC from 0.919 to 0.956
(paired ΔAUC +0.037, 95% CI [+0.010, +0.068]), and at the cut-off it flagged
0 of 75 real photos (Swinv2 alone at 0.99: 5 of 75) while catching more fakes
(80% vs 66%). Do not ship SigLIP. Details and caveats below.

---

## 1. Setup

**Detectors** (all frozen, zero-shot, on one Modal T4):

| key | model | architecture | why it was a candidate |
|---|---|---|---|
| swinv2 | `haywoodsloan/ai-image-detector-deploy` | Swinv2, 256 px | current production model |
| commfor | `OwensLab/commfor-model-384` (MIT) | ViT-S/16, 384 px, 1 logit | Community Forensics, CVPR 2025: trained on 4,803 generators, no DIV2K, FLUX-schnell only in its eval split |
| siglip | `Ateeqq/ai-vs-human-image-detector` (Apache-2.0) | SigLIP, 224 px | drop-in, but training data undisclosed and its model card warns about overfitting |

The candidates came from a search of Hugging Face (every model's existence, licence and
label mapping was verified). Rejected before testing: `Organika/sdxl-detector`
(non-commercial, and Swin like the incumbent), `umm-maybe/AI-image-detector` (2022,
broken config), UnivFD (trained on ProGAN only), and `Bombek1/...siglip-dinov2`
(2.1 GB, training data CC-BY-NC).

**Ensembles:** the mean of the two log-odds ("mean"), the minimum probability, so both
detectors must agree ("min"), and the mean of all three.

**Eval images.** Every image went through the same normalizer as the
`aigc-detect` pilot: JPEG Q95, 4:4:4, a 1024² centre square, no metadata. So no
detector can tell real from fake by container, size or EXIF.

| set | role | real | fake |
|---|---|---|---|
| pilot A0–A8 | threshold tuning + robustness | 25 DIV2K ×9 arms | 25 FLUX.1-schnell ×9 arms |
| **held-out** | **headline numbers** | **75 DIV2K-valid photos the pilot never used** | **74 new FLUX.1-schnell + 74 SDXL-base**, from 74 new prompts |

The pilot's nine arms are: clean, JPEG 90/75/60, resize ×0.7/×0.5, blur σ=0.5,
noise σ=2, and resize→JPEG stacked. Every cut-off below was chosen on the pilot and
applied **unchanged** to the held-out set, so no held-out number is fitted.

## 2. Findings

**B1. The Swinv2 + CommFor ensemble is a significant, paired improvement.** On held-out
images, AUC rose from 0.919 to 0.956. The paired bootstrap resamples images jointly,
so both detectors are compared on the same draws: ΔAUC = +0.037, 95% CI
[+0.010, +0.068], P(Δ ≤ 0) = 0.004. The two models disagree usefully: Spearman ρ = 0.62
between their scores. Each is strong where the other is weak. CommFor is near-perfect
on SDXL (AUC 0.996 vs Swinv2 0.943), and Swinv2 is better on FLUX (0.895 vs 0.853).

**B2. At the product's operating point, the ensemble falsely accuses fewer real photos
and catches more fakes.**

| config | reals flagged "likely AI" | fakes caught |
|---|---|---|
| Swinv2 @ 0.99 (current) | 5/75 = 6.7% [2.9–14.7] | 97/148 = 65.5% [57.6–72.7] |
| Swinv2+CommFor mean @ 0.64 | **0/75 = 0.0% [0.0–4.9]** | **118/148 = 79.7% [72.5–85.4]** |

95% Wilson intervals. The false-positive intervals overlap, so "0%" means below about
5%, not zero. Under the same pilot-tuned cut-off, CommFor alone overshot on held-out
reals (4% → 12% FPR). The ensemble's calibration transferred; the single model's
did not.

**B3. CommFor alone is fragile under compression; the ensemble inherits Swinv2's
robustness.** CommFor's AUC collapses to 0.670 at JPEG Q60 and 0.770 on the stacked
arm (Swinv2: 0.880 / 0.882). The ensemble never falls below Swinv2 on any arm
(Q60: 0.880 = 0.880; all other arms +0.015 to +0.053). But it does not fix the
underlying problem: its catch rate on heavily compressed fakes at the conservative
cut-off is low (24% at Q60, against Swinv2's 36% at 0.99). **This is the trade-off:** on
degraded images the ensemble at 0.64 is more cautious than Swinv2 at 0.99. It also
falsely accuses fewer photos under noise (16% vs 24% on the noise arm), which is the
direction that matters most for a tool whose false positive means accusing someone.

**B4. SigLIP is not detecting generation artefacts.** Its score moves by at most
**0.007** between an image and its JPEG-Q60 copy (mean |Δ| = 0.000 on every arm), while
Swinv2 and CommFor move by 0.09–0.17 exactly where artefacts are destroyed. Its
per-arm AUC is flat at 0.930. A detector that is blind to compression, resizing and
noise is reading *content*: what a FLUX prompt tends to depict, not how it was
rendered. On held-out images with new prompts, it fell to 0.873, the worst of the
three, and on SDXL to 0.839. It is overfit to subject matter, as its model card warns.
Adding it to the ensemble gains a little AUC (all-3 mean: 0.966), but the gain rests
on a content shortcut that will not generalize, so we excluded it.

**B5. Swinv2's margin over a trivial baseline is real on held-out data.** File size
alone (compressibility at fixed Q95) scores AUC 0.811, which is a strong baseline
because DIV2K photos carry more texture than diffusion output. Swinv2 beats it by
+0.108 and the ensemble by +0.145. The pilot found a margin of only +0.02, so on the
larger held-out set that worry is smaller, but the baseline stays in every table.

**B6. Calibration, not ranking, is still the weak point.** The production bands
(0.99 / 0.10) on held-out data: real → 81% "no strong signal", 12% inconclusive, 7%
"likely AI"; fake → 66% / 20% / 14%. So 14% of fakes get the reassuring verdict.
That is the known limit of passive detection, and the UI copy already says a low
score does not clear an image.

## 3. What this does and doesn't show

- **n is small.** 75 real / 148 fake held-out, one real source (DIV2K professional
  photography), two open generators. The CIs are honest but wide. Nothing here covers
  Midjourney, DALL·E 3, GPT-image, Imagen or FLUX-dev, where the literature reports
  18–30% detection for modern generators.
- **DIV2K vs diffusion is an easier contrast than phone photos vs diffusion.** That is
  why the file-size baseline is so strong, and why absolute numbers here are an upper
  bound.
- **Synthetic degradation is not a platform round-trip.** WeChat/WhatsApp re-encoding
  (the `aigc-detect` B arms) is still untested.
- CommFor saw FLUX-schnell only in its eval split. SDXL is almost certainly in its
  training data (4,803 generators), so its 0.996 on SDXL is an in-distribution number.

## 4. Decision and next steps

1. Add CommFor-384 to the Modal backend, and return both per-model scores plus the
   ensemble score. Set the verdict bands for the ensemble: "likely AI" ≥ 0.64 (pilot
   FPR ≤ 4%). For "no strong signal", tune ≤ x on the pilot so that at most ~10% of
   fakes land there, then check it on held-out data.
2. Deploy it as a separate Modal app, point the dev server at it, and switch production
   only after review. The current Modal app also serves VeriDoc's contract endpoint.
3. Grow the held-out set with commercial generators and phone photos before quoting
   any number publicly.

## 5. Reproduce

```bash
.venv/bin/python bench/run.py                                        # generates, normalizes, scores (~1 h, Modal)
/root/projects/aigc-detect/.venv/bin/python bench/analyze.py > bench/results/report-tables.md
/root/projects/aigc-detect/.venv/bin/python bench/figures.py         # paper/figs/*.pdf
```
