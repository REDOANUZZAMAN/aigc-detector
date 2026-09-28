import io
import os

import pytest
from PIL import Image, PngImagePlugin

os.environ.setdefault("MODAL_URL", "https://backend.test")
os.environ.setdefault("MODAL_API_KEY", "backend-key")
os.environ.setdefault("API_KEYS", "test-api-key")


def make_png(text: dict | None = None, size=(256, 256)) -> bytes:
    img = Image.new("RGB", size, (120, 80, 200))
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def make_jpeg(exif: dict | None = None, size=(256, 256)) -> bytes:
    img = Image.new("RGB", size, (30, 140, 90))
    ex = Image.Exif()
    for k, v in (exif or {}).items():
        ex[k] = v
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=ex.tobytes())
    return buf.getvalue()


@pytest.fixture
def png():
    return make_png


@pytest.fixture
def jpeg():
    return make_jpeg
