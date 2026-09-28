"""AIGC detector -- web front end and public API.

Stateless: uploaded images are forwarded to the Modal GPU backend and never
written to disk.
"""
import io
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import metadata

try:  # iPhone photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MODAL_URL = os.environ["MODAL_URL"].rstrip("/")
MODAL_API_KEY = os.environ["MODAL_API_KEY"]
APP_USER = os.environ.get("APP_USER", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")          # empty = public site
API_KEYS = {k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()}
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_HOUR", "30"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "15")) * 1024 * 1024
# Formats the backend decodes natively; anything else is re-encoded to PNG here.
NATIVE = {"JPEG", "PNG", "WEBP"}

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="AIGC Detector", docs_url="/api/docs", redoc_url=None)
basic = HTTPBasic(auto_error=False)
_hits: dict[str, deque] = defaultdict(deque)
_last_warm = 0.0


def client_ip(request: Request) -> str:
    h = request.headers
    return (h.get("cf-connecting-ip")
            or h.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown"))


def gate(request: Request, creds: HTTPBasicCredentials | None = Depends(basic)) -> str:
    """Returns "key" for API-key callers (no rate limit), else "web"."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and API_KEYS:
        if any(secrets.compare_digest(auth[7:].strip().encode(), k.encode()) for k in API_KEYS):
            return "key"
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
    if APP_PASSWORD:
        ok = creds is not None and secrets.compare_digest(
            creds.username.encode(), APP_USER.encode()) and secrets.compare_digest(
            creds.password.encode(), APP_PASSWORD.encode())
        if not ok:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthorized",
                                headers={"WWW-Authenticate": "Basic"})
    return "web"


def rate_limit(request: Request, caller: str) -> None:
    if caller == "key" or RATE_LIMIT <= 0:
        return
    now, q = time.time(), _hits[client_ip(request)]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        retry = int(3600 - (now - q[0])) + 1
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"Rate limit: {RATE_LIMIT} analyses per hour.",
                            headers={"Retry-After": str(retry)})
    q.append(now)


@app.get("/")
def index(_: str = Depends(gate)):
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{MODAL_URL}/healthz")
        backend = r.json() if r.status_code == 200 else {"status": f"http {r.status_code}"}
    except httpx.HTTPError as e:
        backend = {"status": f"unreachable: {type(e).__name__}"}
    return {"status": "ok", "backend": backend}


@app.post("/api/warm")
async def warm(_: str = Depends(gate)):
    """Spin up a GPU container while the user picks a file. Debounced globally."""
    global _last_warm
    if time.time() - _last_warm < 60:
        return {"status": "warm-requested"}
    _last_warm = time.time()
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            await c.post(f"{MODAL_URL}/warm",
                         headers={"Authorization": f"Bearer {MODAL_API_KEY}"})
    except httpx.HTTPError:
        pass
    return {"status": "warm-requested"}


@app.post("/api/analyze")
async def analyze(request: Request, file: UploadFile = File(...), caller: str = Depends(gate)):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // 1024 // 1024} MB).")
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        meta = metadata.inspect(data)
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(415, "Not a supported image (JPEG, PNG, WebP, HEIC, GIF, BMP, TIFF).")

    rate_limit(request, caller)

    payload, name = data, file.filename or "upload"
    if meta["format"] not in NATIVE:
        with Image.open(io.BytesIO(data)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG")
        payload, name = buf.getvalue(), "converted.png"

    t0 = time.time()
    try:
        # 150 s covers a cold GPU start (~60 s) with margin.
        async with httpx.AsyncClient(timeout=150) as c:
            r = await c.post(f"{MODAL_URL}/v1/analyze",
                             headers={"Authorization": f"Bearer {MODAL_API_KEY}"},
                             files={"file": (name, payload)})
    except httpx.TimeoutException:
        raise HTTPException(504, "Detector timed out. The GPU may be starting; try again.")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Detector unreachable ({type(e).__name__}).")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text[:200])
        except ValueError:
            detail = f"backend returned HTTP {r.status_code}"
        raise HTTPException(502 if r.status_code >= 500 else r.status_code, detail)

    result = r.json()
    result["metadata"] = meta
    result["file"] = {"name": file.filename, "bytes": len(data), "format": meta["format"]}
    result["elapsed_s"] = round(time.time() - t0, 2)
    return JSONResponse(result)
