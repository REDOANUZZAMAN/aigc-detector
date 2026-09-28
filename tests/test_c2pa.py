from pathlib import Path

import c2pa
import pytest

from app import c2pa_check, metadata

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return (FIX / name).read_bytes()


@pytest.fixture(scope="module")
def test_trust():
    """Trust the C2PA SDK's test CA, standing in for a real trust-list signer."""
    anchors = (FIX / "test_anchors.pem").read_text()
    return c2pa.Context(c2pa.Settings.from_dict(c2pa_check.trust_settings(anchors)))


def test_no_manifest_returns_none(png):
    assert c2pa_check.verify(png(), "PNG") is None


def test_production_trust_list_rejects_test_cert():
    r = c2pa_check.verify(load("C.jpg"), "JPEG")
    assert r["state"] == "valid" and r["signer"] == "C2PA Signer"


def test_trusted_signer(test_trust):
    r = c2pa_check.verify(load("C.jpg"), "JPEG", test_trust)
    assert r["state"] == "trusted" and not r["errors"]


def test_tampered_file_is_invalid(test_trust):
    r = c2pa_check.verify(load("XCA.jpg"), "JPEG", test_trust)
    assert r["state"] == "invalid" and "assertion.dataHash.mismatch" in r["errors"]


def test_algorithmic_media_is_not_ai():
    # C.jpg's manifest says digitalSourceType=algorithmicMedia (a gradient), not AI.
    assert c2pa_check.verify(load("C.jpg"), "JPEG")["declares_ai"] is False


def test_ai_declaration_flows_into_metadata():
    m = metadata.inspect(load("ai_signed.jpg"))
    assert m["declared_ai"] is True
    assert m["c2pa"]["ai_actions"] == ["created"]
    assert m["c2pa"]["generator"] == "Test Image Generator"
    assert any(f["key"] == "c2pa_ai" for f in m["findings"])
