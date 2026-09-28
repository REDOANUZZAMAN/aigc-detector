# AIGC Detector

Upload an image and find out whether it was AI-generated. You get a classifier score, a 4×4
region heatmap, and a check of the file's provenance metadata (C2PA, IPTC, EXIF, PNG generator
tags).

```
browser ──► FastAPI web app (dokploy, this repo /app) ──► Modal GPU backend (modal_app.py, T4)
             · provenance metadata                          · Swinv2 classifier, full image
             · rate limit / auth                            · same model over a 4×4 tile grid
             · HEIC/BMP/TIFF → PNG                          · weight-0 forensic diagnostics
```

Images are processed in memory and never written to disk.

## Model and how to read the score

- **Classifier:** [`haywoodsloan/ai-image-detector-deploy`](https://huggingface.co/haywoodsloan/ai-image-detector-deploy),
  Swinv2 at 256 px, Apache-2.0. Its P(artificial) is the only input to the verdict.
- **Verdict bands:** `≥ 0.99` likely AI · `≤ 0.10` no strong AI signal · otherwise inconclusive.
  The upper band is high on purpose. In the n=50 pilot (`aigc-detect`, 2026-09-22) the
  clean-set optimal threshold was 0.994, and 20% of real photos scored above 0.5.
- **Heatmap:** every tile is scored independently by the same model. Hot tiles inside a cool
  image point to local edits.
- **Provenance:** reported next to the score, never folded into it. A generator tag is strong
  evidence. Missing metadata is not evidence of anything.
- **Diagnostics:** FFT high-frequency energy, noise-residual uniformity, and JPEG grid
  discontinuity. These carry weight 0.0 and are shown for explainability only.

Known limits: recompression (WeChat, WhatsApp) pulls scores down and noise pushes them up.
Recent generators (FLUX dev, Midjourney v7, Firefly 4) evade public detectors. A low score does
not clear an image.

## API

```bash
curl -H "Authorization: Bearer $API_KEY" -F file=@photo.jpg https://<host>/api/analyze
```

The response has `likelihood`, `verdict{key,label}`, `bands`, `tiles[{row,col,x,y,width,height,score}]`
with normalized coordinates, `indicators[]`, `metadata{declared_ai,camera,findings[]}`,
`file`, and `elapsed_s`. Web callers without a key are rate-limited per IP. OpenAPI docs are at
`/api/docs`.

The Modal backend also serves VeriDoc's `remote_http` contract at `POST /`, so VeriDoc can point
`AIGC_REMOTE_URL` at `https://chinesemahim--aigc-detector-api.modal.run/` without code changes.

## Run

```bash
# backend (one-time: modal secret create aigc-detector-key AIGC_KEY=...)
modal deploy modal_app.py        # if behaviour doesn't change: modal app stop aigc-detector -y, then redeploy

# web app
cp .env.example .env             # fill MODAL_API_KEY
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
set -a; . ./.env; set +a; .venv/bin/uvicorn app.server:app --reload
.venv/bin/pytest -q
```

Deploy: build `Dockerfile` (container port 8000) with the env vars from `.env.example`.
Production runs as a self-managed compose stack at `/root/stacks/aigc-detector` on the
Contabo VPS, routed by Traefik at `https://aigc.rmaikr.top`. It is not on dokploy, so a push
doesn't redeploy. To ship a change: `cd /root/stacks/aigc-detector && docker compose up -d --build`.
The first request after 5 idle minutes cold-starts the GPU (~20–60 s). The page sends
`/api/warm` on load to hide most of that wait.
