"""Second-detector benchmark on Modal.

Scores the same images with the production Swinv2 and candidate detectors, so
the comparison is paired. Also produces the extra eval images that are cheap to
make on Modal: SDXL fakes, and the 75 DIV2K-valid photos the pilot didn't use.
Everything is normalized by aigc-detect's harness (JPEG Q95, 1024² centre
square, no metadata) so no detector can win on container format.

    modal run bench/modal_bench.py     # driven by bench/run.py, not directly
"""
import modal

app = modal.App("aigc-bench")

NORMALIZE = "/root/projects/aigc-detect/harness/normalize.py"
SWIN = "haywoodsloan/ai-image-detector-deploy"
COMMFOR = "OwensLab/commfor-model-384"
SIGLIP = "Ateeqq/ai-vs-human-image-detector"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def download_detectors():
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import timm

    for repo in (SWIN, SIGLIP):
        AutoImageProcessor.from_pretrained(repo)
        AutoModelForImageClassification.from_pretrained(repo)
    hf_hub_download(COMMFOR, "model.safetensors")
    timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)


def download_sdxl():
    import torch
    from diffusers import StableDiffusionXLPipeline
    StableDiffusionXLPipeline.from_pretrained(SDXL, torch_dtype=torch.float16, variant="fp16")


base = (modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.4.1", "torchvision==0.19.1", "transformers==4.44.2",
                     "timm==1.0.9", "safetensors", "huggingface_hub", "pillow==10.4.0",
                     "numpy==1.26.4", "requests"))
det_image = base.run_function(download_detectors, timeout=1800).add_local_file(
    NORMALIZE, "/root/normalize.py")
gen_image = (base.pip_install("diffusers==0.30.3", "accelerate")
             .run_function(download_sdxl, timeout=1800)
             .add_local_file(NORMALIZE, "/root/normalize.py"))


def canon(png_bytes: bytes) -> bytes:
    """Run bytes through the harness normalizer (it works on paths)."""
    import sys, tempfile
    from pathlib import Path
    sys.path.insert(0, "/root")
    from normalize import normalize_file
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in", Path(d) / "out.jpg"
        src.write_bytes(png_bytes)
        normalize_file(src, dst)
        return dst.read_bytes()


@app.function(image=gen_image, gpu="A10G", timeout=3600)
def gen_sdxl(prompts: list[str], seed: int = 1234) -> list[bytes]:
    import io
    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL, torch_dtype=torch.float16, variant="fp16").to("cuda")
    out = []
    for i, p in enumerate(prompts):
        g = torch.Generator("cuda").manual_seed(seed + i)
        img = pipe(p, num_inference_steps=30, guidance_scale=6.0, generator=g,
                   width=1024, height=1024).images[0]
        buf = io.BytesIO()
        img.save(buf, "PNG")
        out.append(canon(buf.getvalue()))
    return out


@app.function(image=det_image, timeout=1800)
def div2k_holdout(used: list[str]) -> dict[str, bytes]:
    """The DIV2K-valid photos the pilot didn't use, normalized."""
    import io, zipfile, requests
    buf = io.BytesIO(requests.get(DIV2K_URL, timeout=900).content)
    out = {}
    with zipfile.ZipFile(buf) as z:
        for n in sorted(z.namelist()):
            stem = n.rsplit("/", 1)[-1].removesuffix(".png")
            if n.endswith(".png") and stem not in used:
                out[stem] = canon(z.read(n))
    return out


@app.cls(image=det_image, gpu="T4", timeout=1800)
class Scorer:
    @modal.enter()
    def load(self):
        import timm, torch
        import torchvision.transforms as T
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        def hf(repo):
            proc = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForImageClassification.from_pretrained(repo).eval().cuda()
            labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
            return proc, model, labels

        self.swin = hf(SWIN)
        self.siglip = hf(SIGLIP)
        print("labels", self.swin[2], self.siglip[2])

        m = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k",
                              pretrained=False, num_classes=1)
        sd = load_file(hf_hub_download(COMMFOR, "model.safetensors"))
        m.load_state_dict({k.removeprefix("vit."): v for k, v in sd.items()})
        self.commfor = m.eval().cuda()
        self.commfor_tf = T.Compose([T.Resize(440), T.CenterCrop(384), T.ToTensor(),
                                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _hf_score(self, bundle, imgs, fake_prefixes):
        import torch
        proc, model, labels = bundle
        idx = next(i for i, l in labels.items() if l.startswith(fake_prefixes))
        with torch.no_grad():
            logits = model(**proc(images=imgs, return_tensors="pt").to("cuda")).logits
        return torch.softmax(logits, -1)[:, idx].cpu().tolist()

    @modal.method()
    def score(self, items: list[tuple[str, bytes]]) -> list[dict]:
        import io, torch
        from PIL import Image
        out = []
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            imgs = [Image.open(io.BytesIO(b)).convert("RGB") for _, b in chunk]
            swin = self._hf_score(self.swin, imgs, ("artificial", "ai", "fake"))
            sig = self._hf_score(self.siglip, imgs, ("ai",))
            with torch.no_grad():
                x = torch.stack([self.commfor_tf(im) for im in imgs]).cuda()
                com = torch.sigmoid(self.commfor(x)).squeeze(1).cpu().tolist()
            out += [{"key": k, "swinv2": s, "commfor": c, "siglip": g}
                    for (k, _), s, c, g in zip(chunk, swin, com, sig)]
        return out
