"""AIGC detector -- GPU inference backend on Modal.

Standalone service extracted from VeriDoc's synthetic-media seam.

    POST /v1/analyze   Authorization: Bearer $AIGC_KEY
                       multipart/form-data: file=<image>
                       -> {"likelihood", "verdict", "tiles", "regions", "indicators", ...}
    POST /             VeriDoc `remote_http` contract (file, page, document_type)
    POST /warm         start a GPU container without running inference
    GET  /healthz

The headline `likelihood` comes from a trained classifier
(haywoodsloan/ai-image-detector-deploy, Swinv2 @ 256px, Apache-2.0,
labels {0: artificial, 1: real}). The heatmap comes from running the same
classifier over a tile grid, so localization is model-derived. `indicators`
beyond the classifier are signal-processing diagnostics with weight 0.0 --
they explain, they do not score.
"""
import os

import modal

app = modal.App("aigc-detector")

MODEL_REPO = "haywoodsloan/ai-image-detector-deploy"
MODEL_VERSION = "swinv2-2024.11"
TILE_GRID = 4           # 4x4 heatmap
REGION_THRESHOLD = 0.55 # tiles above this are reported as regions
MAX_SIDE = 2048         # images are downscaled to this before tiling

# Verdict bands. The classifier is badly calibrated around 0.5: in the n=50
# pilot (aigc-detect, 2026-09-22) the clean-set optimal threshold was 0.994
# and 20% of real photos scored above 0.5. Bands, not a single cut-off.
LIKELY_AI = 0.99
LIKELY_REAL = 0.10


def download_model():
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    AutoImageProcessor.from_pretrained(MODEL_REPO)
    AutoModelForImageClassification.from_pretrained(MODEL_REPO)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "pillow==10.4.0",
        "numpy==1.26.4",
        "fastapi[standard]",
        "python-multipart",
    )
    .run_function(download_model, timeout=900)
)

with image.imports():
    import io

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModelForImageClassification


def verdict_for(p: float) -> dict:
    if p >= LIKELY_AI:
        return {"key": "likely_ai", "label": "Likely AI-generated"}
    if p <= LIKELY_REAL:
        return {"key": "likely_real", "label": "No strong AI signal"}
    return {"key": "inconclusive", "label": "Inconclusive"}


@app.cls(
    image=image,
    gpu="T4",
    timeout=300,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("aigc-detector-key")],
)
class Detector:
    @modal.enter()
    def load(self):
        self.processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_REPO)
        self.model.eval().to("cuda")
        # Resolve which logit index means "generated" from the model's own config.
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.fake_idx = next(
            (i for i, lbl in id2label.items()
             if lbl.startswith(("artificial", "ai", "fake", "generated"))),
            0,
        )

    def _score(self, pil_images: list) -> list[float]:
        """P(artificial) for each image, batched."""
        batch = self.processor(images=pil_images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[:, self.fake_idx]
        return [float(p) for p in probs.cpu()]

    @staticmethod
    def _diagnostics(img) -> list[dict]:
        """Deterministic forensic diagnostics. Explainability only, weight 0."""
        g = np.asarray(img.convert("L"), dtype=np.float64)
        if g.size == 0:
            return []
        out = []

        # 1. High-frequency spectral energy: generated imagery is often
        #    smoother in the upper frequency bands than a camera sensor.
        f = np.fft.fftshift(np.abs(np.fft.fft2(g)))
        h, w = f.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = max(float(r.max()), 1.0)
        total = float(f.sum()) or 1.0
        hi = float(f[r > 0.55 * rmax].sum()) / total
        out.append({
            "key": "spectral_high_freq_ratio",
            "label": "High-frequency spectral energy",
            "score": round(hi, 6), "weight": 0.0,
            "note": "Share of FFT magnitude beyond 55% of the Nyquist radius. "
                    "Diagnostic only; does not affect the score.",
        })

        # 2. Noise-residual uniformity: synthetic images tend to carry a more
        #    even noise floor than optics + sensor + compression do.
        res = g[1:, 1:] - 0.5 * (g[:-1, 1:] + g[1:, :-1])
        bs = 32
        bh, bw = res.shape[0] // bs, res.shape[1] // bs
        if bh and bw:
            blocks = res[:bh * bs, :bw * bs].reshape(bh, bs, bw, bs).swapaxes(1, 2)
            stds = blocks.reshape(bh * bw, -1).std(axis=1)
            uni = float(stds.std() / (stds.mean() + 1e-9))
            out.append({
                "key": "noise_residual_uniformity",
                "label": "Noise-residual uniformity",
                "score": round(uni, 6), "weight": 0.0,
                "note": "Coefficient of variation of per-block noise. Lower means a "
                        "more uniform noise floor. Diagnostic only.",
            })

        # 3. JPEG blockiness: a discontinuity at the 8px grid indicates a
        #    recompression history.
        if g.shape[0] > 16 and g.shape[1] > 16:
            d = np.abs(np.diff(g, axis=1))
            on = d[:, 7::8].mean() if d[:, 7::8].size else 0.0
            off = d.mean() or 1e-9
            out.append({
                "key": "jpeg_grid_discontinuity",
                "label": "JPEG 8px grid discontinuity",
                "score": round(float(on / off), 6), "weight": 0.0,
                "note": "Ratio of gradient energy on the JPEG block boundary to the "
                        "image average. >1 suggests recompression. Diagnostic only.",
            })
        return out

    @modal.method()
    def warm(self) -> bool:
        return True

    @modal.method()
    def analyze(self, image_bytes: bytes, grid: int = TILE_GRID) -> dict:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        orig_w, orig_h = img.size
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        W, H = img.size

        tiles, boxes = [], []
        tw, th = W // grid, H // grid
        if tw >= 32 and th >= 32:
            for gy in range(grid):
                for gx in range(grid):
                    box = (gx * tw, gy * th, (gx + 1) * tw, (gy + 1) * th)
                    tiles.append(img.crop(box))
                    boxes.append((gx, gy, box))

        scores = self._score([img] + tiles)
        p, tile_scores = scores[0], scores[1:]

        # Tile coordinates are returned normalized to [0, 1] so the client can
        # overlay them on whatever size it renders the image at.
        tile_out = [
            {"col": gx, "row": gy,
             "x": b[0] / W, "y": b[1] / H,
             "width": (b[2] - b[0]) / W, "height": (b[3] - b[1]) / H,
             "score": round(s, 6)}
            for (gx, gy, b), s in zip(boxes, tile_scores)
        ]

        indicators = [{
            "key": "classifier_confidence",
            "label": "Trained classifier confidence",
            "score": round(p, 6), "weight": 1.0,
            "note": f"P(artificial) from {MODEL_REPO} over the full image. "
                    f"This is the signal behind the score.",
        }] + self._diagnostics(img)

        return {
            "likelihood": p,
            "verdict": verdict_for(p),
            "bands": {"likely_ai": LIKELY_AI, "likely_real": LIKELY_REAL},
            "model": MODEL_REPO,
            "model_version": MODEL_VERSION,
            "image": {"width": orig_w, "height": orig_h,
                      "analyzed_width": W, "analyzed_height": H},
            "grid": grid if tiles else 0,
            "tiles": tile_out,
            "indicators": indicators,
        }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aigc-detector-key")],
    timeout=300,
)
@modal.asgi_app()
def api():
    import secrets as _secrets

    from fastapi import FastAPI, File, Form, Header, UploadFile
    from fastapi.responses import JSONResponse

    web = FastAPI(title="AIGC detector backend")

    def authorized(header: str) -> bool:
        expected = os.environ["AIGC_KEY"]
        return header.startswith("Bearer ") and _secrets.compare_digest(
            header[7:].strip().encode(), expected.encode())

    unauthorized = lambda: JSONResponse({"detail": "Unauthorized"}, status_code=401)

    @web.post("/v1/analyze")
    async def _analyze(
        file: UploadFile = File(...),
        grid: int = Form(TILE_GRID),
        authorization: str = Header(default=""),
    ):
        if not authorized(authorization):
            return unauthorized()
        grid = max(2, min(grid, 6))
        data = await file.read()
        try:
            return JSONResponse(await Detector().analyze.remote.aio(data, grid))
        except Exception as e:  # undecodable image etc.
            return JSONResponse({"detail": f"Analysis failed: {e}"}, status_code=422)

    @web.post("/")
    async def _veridoc(
        file: UploadFile = File(...),
        page: int = Form(1),
        document_type: str = Form(""),
        authorization: str = Header(default=""),
    ):
        """VeriDoc `remote_http` contract, so VeriDoc can point here unchanged."""
        if not authorized(authorization):
            return unauthorized()
        r = await Detector().analyze.remote.aio(await file.read(), 3)
        W, H = r["image"]["analyzed_width"], r["image"]["analyzed_height"]
        regions = [
            {"x": round(t["x"] * W), "y": round(t["y"] * H),
             "width": round(t["width"] * W), "height": round(t["height"] * H),
             "score": t["score"]}
            for t in r["tiles"] if t["score"] >= REGION_THRESHOLD
        ]
        p = r["likelihood"]
        if p >= 0.5:
            summary = (f"Classifier reports {p:.0%} likelihood of generated or "
                       f"AI-edited imagery on page {page}.")
        else:
            summary = (f"No significant synthetic-media indicators on page {page} "
                       f"({p:.0%} likelihood).")
        if regions:
            summary += f" {len(regions)} of {len(r['tiles'])} page regions scored above threshold."
        return JSONResponse({
            "likelihood": p,
            "model": MODEL_REPO,
            "model_version": MODEL_VERSION,
            "provider_label": "Swinv2 synthetic-media classifier (self-hosted)",
            "summary": summary,
            "indicators": r["indicators"],
            "regions": regions,
            "document_type": document_type,
            "page": page,
        })

    @web.post("/warm")
    async def _warm(authorization: str = Header(default="")):
        if not authorized(authorization):
            return unauthorized()
        await Detector().warm.spawn.aio()
        return {"status": "warming"}

    @web.get("/healthz")
    def _healthz():
        return {"status": "ok", "model": MODEL_REPO, "model_version": MODEL_VERSION}

    return web
