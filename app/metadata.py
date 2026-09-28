"""File-level provenance checks, run on the VPS before GPU inference.

Metadata is weak evidence in one direction only: a generator tag or a C2PA
manifest saying "AI" is a strong positive, but its *absence* proves nothing --
every social platform strips it. Findings are therefore reported separately
from the classifier score and never folded into it.
"""
import io
import re

from PIL import ExifTags, Image

# Substrings that identify a generator when they appear in metadata text.
GENERATORS = {
    "midjourney": "Midjourney",
    "dall-e": "DALL·E",
    "dall·e": "DALL·E",
    "openai": "OpenAI",
    "chatgpt": "ChatGPT",
    "firefly": "Adobe Firefly",
    "stable diffusion": "Stable Diffusion",
    "stablediffusion": "Stable Diffusion",
    "sdxl": "Stable Diffusion XL",
    "comfyui": "ComfyUI",
    "automatic1111": "AUTOMATIC1111",
    "novelai": "NovelAI",
    "flux": "FLUX",
    "imagen": "Google Imagen",
    "gemini": "Google Gemini",
    "leonardo.ai": "Leonardo.Ai",
    "ideogram": "Ideogram",
    "dreamstudio": "DreamStudio",
    "bing image creator": "Bing Image Creator",
    "designer": None,  # too generic alone; only counted with "microsoft"
}

# IPTC DigitalSourceType values that declare synthetic content.
IPTC_AI = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia",
           "algorithmicmedia")

# PNG text keys written by local diffusion front-ends.
SD_PNG_KEYS = {"parameters": "AUTOMATIC1111 / Forge generation parameters",
               "prompt": "ComfyUI prompt graph",
               "workflow": "ComfyUI workflow",
               "dream": "InvokeAI parameters",
               "sd-metadata": "InvokeAI metadata",
               "invokeai_metadata": "InvokeAI metadata"}


def _match_generator(text: str) -> str | None:
    t = text.lower()
    if "microsoft" in t and "designer" in t:
        return "Microsoft Designer"
    for needle, name in GENERATORS.items():
        if name and needle in t:
            # "flux" is a common English word; require a word boundary.
            if needle == "flux" and not re.search(r"\bflux(\.1)?\b", t):
                continue
            return name
    return None


def inspect(data: bytes) -> dict:
    """Return {"format", "findings": [...], "declared_ai": bool, "camera": str|None}."""
    findings: list[dict] = []
    declared_ai = False
    camera = None

    def add(key, label, value, signal):
        findings.append({"key": key, "label": label, "value": value, "signal": signal})

    img = Image.open(io.BytesIO(data))
    fmt = img.format or "unknown"

    # --- C2PA / Content Credentials (JUMBF box) -------------------------------
    if b"c2pa" in data and (b"jumb" in data or b"caBX" in data):
        add("c2pa", "Content Credentials (C2PA)",
            "A C2PA manifest is embedded. Verify it at contentcredentials.org/verify.",
            "info")
        if any(s.encode() in data.lower() for s in IPTC_AI):
            declared_ai = True
            add("c2pa_ai", "C2PA declares AI", "Manifest records trained-algorithmic media.", "ai")

    # --- XMP (IPTC digital source type, creator tool) -------------------------
    xmp = b""
    start = data.find(b"<x:xmpmeta")
    if start != -1:
        end = data.find(b"</x:xmpmeta>", start)
        xmp = data[start:end + 12] if end != -1 else data[start:start + 65536]
    if xmp:
        low = xmp.decode("utf-8", "ignore").lower()
        if any(s in low for s in IPTC_AI) and not declared_ai:
            declared_ai = True
            add("iptc_source", "IPTC digital source type",
                "trainedAlgorithmicMedia (declared AI-generated)", "ai")
        m = re.search(r'creatortool[=>"\s]+([^"<]+)', low)
        if m:
            gen = _match_generator(m.group(1))
            add("xmp_tool", "XMP creator tool", m.group(1).strip()[:120], "ai" if gen else "neutral")
            declared_ai |= bool(gen)

    # --- PNG text chunks -------------------------------------------------------
    if fmt == "PNG":
        for k, v in (img.info or {}).items():
            if not isinstance(v, str):
                continue
            kl = k.lower()
            if kl in SD_PNG_KEYS:
                declared_ai = True
                add(f"png_{kl}", SD_PNG_KEYS[kl], v[:300] + ("…" if len(v) > 300 else ""), "ai")
            elif kl in ("software", "description", "comment", "title", "source", "author"):
                gen = _match_generator(v)
                declared_ai |= bool(gen)
                add(f"png_{kl}", f"PNG {k}", v[:200], "ai" if gen else "neutral")

    # --- EXIF ------------------------------------------------------------------
    try:
        exif = img.getexif()
    except Exception:
        exif = {}
    if exif:
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        make, model = str(tags.get("Make", "")).strip(), str(tags.get("Model", "")).strip()
        if make or model:
            camera = " ".join(x for x in (make, model) if x)
            add("exif_camera", "Camera", camera, "camera")
        for t in ("Software", "ImageDescription", "Artist", "UserComment"):
            v = tags.get(t)
            if isinstance(v, bytes):
                v = v.decode("utf-8", "ignore")
            if v and str(v).strip():
                gen = _match_generator(str(v))
                declared_ai |= bool(gen)
                add(f"exif_{t.lower()}", f"EXIF {t}", str(v).strip()[:200],
                    "ai" if gen else "neutral")
        try:
            sub = exif.get_ifd(0x8769)  # Exif sub-IFD
            dt = sub.get(0x9003)        # DateTimeOriginal
            if dt:
                add("exif_datetime", "Captured", str(dt), "camera")
        except Exception:
            pass

    if not findings:
        add("none", "No provenance metadata",
            "Nothing embedded. Common for screenshots and anything re-shared through "
            "social apps; says nothing either way.", "neutral")

    return {"format": fmt, "findings": findings, "declared_ai": declared_ai, "camera": camera}
