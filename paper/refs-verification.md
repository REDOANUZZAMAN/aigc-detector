# refs.bib verification log

Built 2026-09-28 for the AIGC-detection review paper. Source list:
`/root/projects/aigc-detect/docs/literature-review.md` (sections 2-7), plus the
classics requested by the author.

## Method

- **arXiv metadata:** the arXiv export API (`export.arxiv.org/api/query`) returned
  HTTP 406 from this VPS, so titles, authors and dates came from the
  `citation_*` meta tags on each `arxiv.org/abs/<id>` page. Those tags carry the same
  metadata the API does.
- **Venues, pages, DOIs:** Crossref API (IEEE, AAAI, Springer, Elsevier, NeurIPS 2025),
  CVF Open Access BibTeX blocks (CVPR 2025, CVPR 2026, CVPRW 2026), PMLR pages (ICML),
  papers.nips.cc (NeurIPS 2023) and proceedings.iclr.cc (ICLR 2024/2025).
  DBLP timed out or rate-limited (429), and OpenReview returned 403 or a bot check, so
  the ViT and DINOv2 venues come from their OpenReview forum IDs as shown in search
  results and from the DINOv2 PDF header ("Published in TMLR (01/2024)").
- **Numbers:** read in full text: arXiv HTML (`arxiv.org/html/<id>`, specific versions
  where it mattered) or the PDF through `pdftotext` (NTIRE 2026).
- **HF models, GitHub repos:** Hugging Face `/api/models/<id>` and `gh api repos/<r>`.
- Where IEEE (Crossref) and CVF page numbers differ, the IEEE/DOI pages are used.

## Claim verification table

| # | Claim in literature review | Status | Where |
|---|---|---|---|
| 1 | NPR F1 82.66 -> 33.98 under mild JPEG [abstract] | **Not found.** Not in the NPR paper, the Mahara review, AIDE, STAL, S2F-Net, SSAFE, MIRROR, AgentFoX, AI-GenBench, 2411.13553, 2407.19553, or six other candidate papers (2505.14359, 2501.15253, 2605.24306, 2504.13726, 2512.12982, 2505.12335). Web searches for the number pair also found nothing. **Don't use it until you have a source.** | none |
| 2 | STAL 98.62 -> 96.24 (2.38 pt drop) from Q=100 to Q=60 [abstract] | **Confirmed.** Balanced accuracy on GenImage. Note: the exact numbers are in the appendix, not the abstract. | 2605.22751 App. D.2 "More Robustness Analysis"; the 2.38-pt drop is also in Sec. 4.3 / Fig. 4 |
| 3 | Bias-free training: ResNet50 71.68 -> 82.74, Swin-T 74.09 -> 85.83, attributed to **2412.17671** [abstract] | **Differs: misattributed.** The numbers are not in 2412.17671 (v1 or v2). They come from **arXiv 2403.17608, Grommelt et al., "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets"** (ECCV 2024 Workshops, LNCS). They are average cross-generator accuracy on GenImage, "Total" row, raw vs JPEG/size-constrained training subsets (+11.06 / +11.74). Cite `grommelt2024fakeorjpeg`, not `guillaro2025biasfree`. | 2403.17608 Table 4, Sec. 4.3 |
| 4 | AIDE +3.5% on AIGCDetectBenchmark, +4.6% on GenImage over prior SOTA [abstract] | **Confirmed.** Prior SOTA = PatchCraft in both cases. | 2406.19435 Abstract; Sec. 5.2 (Tables 3, 4) |
| 5 | Chameleon: PatchCraft 82.30% on GenImage | **Confirmed.** Mean accuracy, trained on SD v1.4. | 2406.19435 Table 4 |
| 6 | Chameleon: PatchCraft 55.70% | **Confirmed**, but only for the **trained-on-All-GenImage** row. PatchCraft is 53.76 (ProGAN-trained) and 56.32 (SD v1.4-trained). | 2406.19435 Table 5 |
| 7 | Chameleon: CNNSpot 56.94% | **Confirmed** for the **ProGAN-trained** row. It is 60.11 (SD1.4) and 60.89 (All GenImage). | Table 5 |
| 8 | Chameleon: FreDect 55.62% | **Confirmed** for the **ProGAN-trained** row. It is 56.86 (SD1.4) and 57.22 (All GenImage). | Table 5 |
| 9 | Chameleon: AIDE 63.89% | **Differs by version.** 63.89 is the All-GenImage row in **arXiv v2**. The current **v3** gives **65.77** (All GenImage), 62.60 (SD1.4), 58.37 (ProGAN). The v2 values are 56.45 and 61.10 for the other two rows. | Table 5 (v2 vs v3) |
| 9a | (Presentation issue with rows 6-9) | The review's Chameleon table mixes training regimes: CNNSpot and FreDect come from ProGAN-trained rows, PatchCraft and AIDE from All-GenImage rows. For a like-for-like comparison, use one row. On All GenImage (v3): CNNSpot 60.89, FreDect 57.22, PatchCraft 55.70, AIDE 65.77. | Table 5 |
| 10 | UnivFD 99.81% on ProGAN vs 50.75% on DALL-E 2 | **Confirmed.** Note that the source is the **AIDE paper's** re-evaluation on AIGCDetectBenchmark (ProGAN training), not the UnivFD paper, so cite `yan2025aide` for it. | 2406.19435 Table 3 |
| 11 | Chameleon 11,170 fake / 14,863 real | **Confirmed** | 2406.19435 Table 1 |
| 12 | All Chameleon fakes passed a human "Turing Test" | **Confirmed** | 2406.19435 Sec. 1 / Sec. 3 |
| 13 | NTIRE 2026: 108,750 real / 185,750 fake / 42 generators / 36 transformations | **Confirmed** | 2604.11487 Abstract; Sec. 2.1 |
| 14 | NTIRE: ~12M images filtered from CC12M / CommonPool / RedCaps; Nano Banana 2, SeeDream 5 Lite | **Confirmed** | Sec. 2.1 |
| 15 | NTIRE: 1-5 random consecutive distortions; primary = robust ROC AUC, secondary = clean ROC AUC | **Confirmed** | Sec. 2.1, 2.2 |
| 16 | NTIRE: 511 registered, 20 teams submitted | **Confirmed** | Abstract |
| 17 | MICV: DINOv3 ensemble, focal loss gamma 2.0 / alpha 0.5, 32x A100 | **Confirmed** (8 h / 10 epochs not re-checked) | Sec. 3.1 |
| 18 | Ant International: 2x DINOv3-7B = 14B, B200, 2.21 img/s, 78.25 GB on one A100 | **Confirmed** | Sec. 3.2 |
| 19 | 2602.07814: 16 methods / 23 variants / 12 datasets / 2.6M images / 291 generators | **Confirmed.** The paper also says results use subsampled, balanced splits of the ~2.6M images. | Abstract; Sec. 3, 3.2 |
| 20 | 2602.07814: Spearman rho 0.01-0.87 | **Confirmed** (median 0.52) | Abstract; Sec. 4.3 (Fig. 4) |
| 21 | 2602.07814: best Community-Forensics 75% vs worst 37.5% (37-pt spread) | **Confirmed.** The worst is AIGCDetectBenchmark_CNNSpot. Caveat: Table 3 shows CommFor at 0.780 mean "after excluding CF v0.2 and v0.3", while the text says 75.0%. | Abstract; Sec. 4.1-4.2, Table 3 |
| 22 | 2602.07814: training data explains 20-60% of variance within architectures | **Confirmed** | Abstract; Sec. 5.1 |
| 23 | 2602.07814: Flux Dev / Firefly v4 / MJ v7 detected at 18-30% | **Confirmed.** Average accuracy across all methods. | Abstract; Sec. 1.2 |
| 24 | 2602.07814: no degradations, zero-shot, fixed 0.5 threshold, AUC-accuracy r = 0.82 | **Confirmed.** Pearson r = 0.82. The paper only resizes and normalizes. | Sec. 3.3 "Zero-shot constraint"; Sec. 4.5, Fig. 6 |
| 25 | GenImage 1,331,167 real / 1,350,000 fake; 8 generators | **Confirmed** | 2306.08571 Table 1; Sec. 3 |
| 26 | Community Forensics: **2.4M** images, 4,803 models | **Differs: 2.7M images.** 4,803 models is correct. arXiv v1, v2, the CVPR 2025 open-access abstract and the HF dataset card all say 2.7M. | 2411.04125 Abstract, Table 1 |
| 27 | anniketkumar/aigc-detect: AUROC 0.9810 clean -> 0.9710 robust | **Confirmed** (baseline checkpoint) | repo README, composite-score table |
| 28 | anniketkumar/aigc-detect: "TPR**@5%FPR**" 53.75% (FLUX.1-dev) to 88.65% (MidJourney) | **Differs: the metric is TPR@1%FPR.** These are clean TPR@1% on the *aug* checkpoint. At 5% FPR the values are 0.8375 (FLUX.1-dev) and 0.9760 (MidJourney). | README finding 2; WRITEUP.md; results/tpr_analysis_aug/report.md |
| 29 | SID_Set: real 100% JPEG, fake 100% PNG, 1024x1024, ICC profiles; container rule = 100.00% | **Confirmed as the repo's claim.** This is an unpublished audit in the repo, not peer-reviewed. | repo README |

## Metadata caveats

- **Tree-Ring (Wen et al.):** the NeurIPS 2023 proceedings title is "Tree-**Rings**
  Watermarks: Invisible Fingerprints for Diffusion Images". The arXiv title differs.
  The bib uses the proceedings title and records the arXiv title in `note`.
- **SynthID:** there is no peer-reviewed SynthID paper. The bib cites the arXiv
  technical report "SynthID-Image: Image watermarking at internet scale" (2510.09263).
- **ForensicHub:** confirmed in NeurIPS 38 (2025) through Crossref (DOI 10.52202/085713-2167).
  Crossref lists it without a track label. The D&B track label comes from the arXiv
  comment ("NeurIPS 2025 DB Track Paper").
- **GenImage:** the NeurIPS 2023 **Datasets and Benchmarks** track is confirmed by the
  proceedings URL (`...-Abstract-Datasets_and_Benchmarks.html`).
- **Mirage (2510.03840):** the arXiv comment says "ACM MM'25, MALLM Workshop". No
  proceedings record was found in Crossref, so it is cited as arXiv with the comment in `note`.
- **SSP (2402.01123):** no peer-reviewed venue found, so it is cited as arXiv.
- **PatchCraft / AIGCDetectBenchmark (2311.12397):** no venue confirmed, so it is cited
  as arXiv.
- **Preprint-only entries** (no venue confirmed as of 2026-09-28): 2411.13553,
  2605.22751, 2601.12313, 2606.08634, 2602.02222, 2603.23115, 2602.07814,
  2508.10104 (DINOv3), 2510.09263.
- **2602.07814 authors:** arXiv shows "Hatsanai (Neo) Tiangratanakul" and
  "Tsang (Dennis) Ng". The nicknames are dropped in the bib.
- **C2PA:** version 2.4 (April 2026) is the latest listed on spec.c2pa.org
  (version history section 5.3.1).
- **IPTC:** year = 2024 is the latest concept-modification date on the vocabulary page
  (`trainedAlgorithmicMedia` modified 2024-10-23). The page carries no edition number.
- **FLUX.1:** there is no paper. The bib cites the official Black Forest Labs GitHub repo
  (created 2024-08-01). The HF FLUX.1-dev card was created 2024-07-31.
- **HF model years** come from each repo's `createdAt`: haywoodsloan 2024-11-16,
  OwensLab 2025-08-28 (tags arXiv 2411.04125), Ateeqq 2025-03-30. The authors are HF
  account names, because no personal names are given.
- **Not given entries** (mentioned in the review, but too generic or not citable):
  Nano Banana 2, SeeDream 5 Lite, eva02 / siglip2 backbones, CC12M / CommonPool / RedCaps,
  and the WeChat / Weibo / WhatsApp platforms.
- **Dropped:** nothing from the requested list. Every requested item was confirmed.
  Added beyond the list because the review relies on them: PatchCraft
  (AIGCDetectBenchmark), SIDA (SID-Set), DINOv3, Grommelt et al. (the true source of
  claim 3), and the three GitHub repos.

## Key index (49 entries)

### Surveys / position
- `mahara2026review`: Mahara & Rishe, comprehensive review of AIGI detection methods (Computer Science Review 2026; arXiv 2502.15176). Source of the taxonomy.
- `guo2024passive`: Guo et al., passive detectors vs watermarking comparison (arXiv 2411.13553).

### Spatial / CNN-artifact
- `wang2020cnnspot`: CNNSpot. ProGAN-trained ResNet generalizes to other CNN generators (CVPR 2020).
- `tan2024npr`: NPR. Neighboring-pixel relationships from up-sampling (CVPR 2024).
- `chen2024ssp`: SSP. A single simple patch's noise pattern is enough (arXiv 2402.01123).
- `zhong2023patchcraft`: PatchCraft. Texture-patch detector; introduces AIGCDetectBenchmark (arXiv 2311.12397).

### Frequency
- `frank2020fredect`: FreDect. DCT-spectrum deepfake detection (ICML 2020).
- `li2026stal`: STAL. Spectral tail auxiliary learning, JPEG-robust (arXiv 2605.22751).
- `hu2026s2fnet`: S^2F-Net. Spatial-spectral fusion for cross-model detection (arXiv 2601.12313).
- `corvi2023detection`: Corvi et al., detecting diffusion-model images; spectral fingerprints (ICASSP 2023).

### Foundation-model features
- `ojha2023univfd`: UnivFD. Frozen CLIP ViT-L/14 + linear head (CVPR 2023).
- `tan2025c2pclip`: C2P-CLIP. Category common prompt injected into CLIP (AAAI 2025).
- `lee2026ssafe`: SSAFE. Frozen vision encoders baseline (arXiv 2606.08634).
- `wang2026ppmclip`: PPM-CLIP. Probabilistic prompt modeling (CVPR 2026).
- `derosa2024clipadv`: Adversarial robustness of CLIP-based detectors (WIFS 2024).

### Reconstruction
- `wang2023dire`: DIRE. Diffusion reconstruction error (ICCV 2023).
- `liu2026mirror`: MIRROR. Manifold reference reconstruction (arXiv 2602.02222).

### Hybrid / explainable
- `yan2025aide`: AIDE + Chameleon. "A Sanity Check for AI-generated Image Detection" (ICLR 2025).
- `sharma2025mirage`: Mirage. LVLM-based artifact explanation (arXiv 2510.03840).
- `yu2026agentfox`: AgentFoX. LLM-agent-guided fusion with explanations (arXiv 2603.23115).

### Benchmarks / datasets / audits
- `zhu2023genimage`: GenImage. Million-scale benchmark, 8 generators (NeurIPS 2023 D&B).
- `park2025commfor`: Community Forensics. 2.7M images from 4,803 generators (CVPR 2025).
- `pellegrini2025aigenbench`: AI-GenBench. Ongoing temporal benchmark (IJCNN 2025).
- `ntire2026robust`: NTIRE 2026 Robust AIGI Detection in the Wild challenge report (CVPRW 2026).
- `du2025forensichub`: ForensicHub. Unified all-domain forensics benchmark and codebase (NeurIPS 2025 D&B).
- `ren2026outofbox`: Zero-shot audit of 16 open detectors on 12 datasets (arXiv 2602.07814).
- `grommelt2024fakeorjpeg`: "Fake or JPEG?". JPEG/size biases in GenImage; source of the ResNet50/Swin-T gains (ECCV 2024 Workshops).
- `guillaro2025biasfree`: B-Free. Bias-free training paradigm (CVPR 2025).
- `yang2026calibrated`: Calibration lets detectors reach SOTA accuracy (AAAI 2026).
- `huang2025sida`: SIDA. Introduces SID-Set social-media deepfake dataset (CVPR 2025).
- `agustsson2017div2k`: DIV2K. High-quality real-image dataset (CVPRW 2017).

### Backbones
- `dosovitskiy2021vit`: ViT (ICLR 2021).
- `liu2022swinv2`: Swin Transformer V2. The backbone of our baseline (CVPR 2022).
- `radford2021clip`: CLIP (ICML 2021).
- `zhai2023siglip`: SigLIP (ICCV 2023).
- `oquab2024dinov2`: DINOv2 (TMLR 2024).
- `simeoni2025dinov3`: DINOv3. The backbone of the NTIRE 2026 winners (arXiv 2508.10104).

### Generators
- `podell2024sdxl`: SDXL (ICLR 2024).
- `blackforestlabs2024flux`: FLUX.1 model family, official repo (@misc).

### Watermarking / provenance
- `wen2023treering`: Tree-Ring watermarks for diffusion images (NeurIPS 2023).
- `gowal2025synthid`: SynthID-Image technical report (arXiv 2510.09263).
- `c2pa2026spec`: C2PA Technical Specification v2.4 (@misc).
- `iptc2024digitalsourcetype`: IPTC Digital Source Type vocabulary (@misc).

### Pretrained checkpoints
- `haywoodsloan2024aidetector`: haywoodsloan/ai-image-detector-deploy. Swinv2 baseline, Apache-2.0.
- `owenslab2025commfor384`: OwensLab/commfor-model-384. Community Forensics checkpoint, MIT.
- `ateeqq2025aivshuman`: Ateeqq/ai-vs-human-image-detector, Apache-2.0.

### Repos / lists
- `kumar2026aigcdetect`: anniketkumar/aigc-detect. 19-cell synthetic transform harness and SID_Set container-leak audit.
- `antresearch2026awesome`: ant-research Awesome-AIGC-Image-Video-Detection list.
- `graydove2024awesome`: graydove Awesome-AIGC-Image-Detection list.

Code repos verified to exist (and cited in `note` fields): WisconsinAIVision/UniversalFakeDetect,
shilinyan99/AIDE, chuangchuangtan/C2P-CLIP-DeepfakeDetection,
bcmi/SSP-AI-Generated-Image-Detection, chuangchuangtan/NPR-DeepfakeDetection.
