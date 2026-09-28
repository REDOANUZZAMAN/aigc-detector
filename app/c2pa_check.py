"""C2PA / Content Credentials verification.

Reads the embedded manifest store with the official c2pa library, checks the
signature against the C2PA trust list plus the contentcredentials.org known-
certificate list (both vendored in app/trust/), and reports who signed it and
whether any manifest in the edit history declares AI-generated content.

Offline by design: remote manifests and OCSP are never fetched, so a crafted
upload can't make the server issue requests.
"""
import io
import json
from pathlib import Path

try:
    import c2pa
except ImportError:  # the byte sniff in metadata.py still runs
    c2pa = None

TRUST = Path(__file__).parent / "trust"
# IPTC digital source types that mean a trained model made the pixels. Plain
# "algorithmicMedia" (fractals, gradients) and "compositeSynthetic" don't.
AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia")
MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
        "HEIF": "image/heic", "TIFF": "image/tiff", "GIF": "image/gif", "AVIF": "image/avif"}

_settings = None


def trust_settings(anchors: str | None = None) -> dict:
    if anchors is None:
        anchors = "".join((TRUST / f).read_text()
                          for f in ("c2pa-trust-list.pem", "cc-anchors.pem"))
    return {
        "verify": {"verify_trust": True, "remote_manifest_fetch": False, "ocsp_fetch": False},
        "trust": {"trust_anchors": anchors,
                  "allowed_list": (TRUST / "cc-allowed.sha256.txt").read_text(),
                  "trust_config": (TRUST / "store.cfg").read_text()},
    }


def _context():
    global _settings
    if _settings is None:
        _settings = trust_settings()
    return c2pa.Context(c2pa.Settings.from_dict(_settings))


def _ai_actions(manifest: dict) -> list[str]:
    out = []
    for a in manifest.get("assertions", []):
        if not a.get("label", "").startswith("c2pa.actions"):
            continue
        for act in (a.get("data") or {}).get("actions", []):
            src = str(act.get("digitalSourceType", "")).lower()
            if src.rsplit("/", 1)[-1] in AI_SOURCE_TYPES:
                out.append(act.get("action", "unknown").removeprefix("c2pa."))
    return out


def _generator(manifest: dict) -> str | None:
    info = manifest.get("claim_generator_info") or []
    names = [i.get("name") for i in info if i.get("name") and i.get("name") != "c2pa-rs"]
    return ", ".join(names) or manifest.get("claim_generator")


def verify(data: bytes, fmt: str, context=None) -> dict | None:
    """None if there's no manifest (or the library is missing), else a summary."""
    if c2pa is None:
        return None
    try:
        with c2pa.Reader(MIME.get(fmt, ""), io.BytesIO(data),
                         context=context or _context()) as r:
            store = json.loads(r.json())
    except Exception:  # ManifestNotFound, unsupported format, corrupt JUMBF
        return None

    manifests = store.get("manifests") or {}
    active = manifests.get(store.get("active_manifest"), {})
    sig = active.get("signature_info") or {}
    errors = [s.get("code") for s in store.get("validation_status") or []
              if s.get("code") != "signingCredential.untrusted"]
    state = {"Trusted": "trusted", "Valid": "valid"}.get(store.get("validation_state"),
                                                         "invalid")
    if errors:
        state = "invalid"

    active_ai = _ai_actions(active)
    history_ai = [m for label, m in manifests.items()
                  if label != store.get("active_manifest") and _ai_actions(m)]
    return {
        "state": state,
        "signer": sig.get("common_name") or sig.get("issuer"),
        "issuer": sig.get("issuer"),
        "signed_at": sig.get("time"),
        "generator": _generator(active),
        "declares_ai": bool(active_ai or history_ai),
        "ai_actions": active_ai,
        "ai_in_history": bool(history_ai),
        "manifests": len(manifests),
        "errors": errors,
    }
