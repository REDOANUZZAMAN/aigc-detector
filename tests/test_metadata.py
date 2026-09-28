from app import metadata


def test_plain_png_has_no_provenance(png):
    r = metadata.inspect(png())
    assert r["format"] == "PNG"
    assert not r["declared_ai"]
    assert [f["key"] for f in r["findings"]] == ["none"]


def test_a1111_parameters_declare_ai(png):
    r = metadata.inspect(png({"parameters": "a cat, Steps: 20, Sampler: Euler a"}))
    assert r["declared_ai"]
    assert any(f["key"] == "png_parameters" and f["signal"] == "ai" for f in r["findings"])


def test_comfyui_workflow_declares_ai(png):
    assert metadata.inspect(png({"workflow": "{}"}))["declared_ai"]


def test_png_software_generator(png):
    assert metadata.inspect(png({"Software": "Adobe Firefly"}))["declared_ai"]
    assert not metadata.inspect(png({"Software": "GIMP 2.10"}))["declared_ai"]


def test_flux_needs_word_boundary():
    assert metadata._match_generator("FLUX.1 schnell") == "FLUX"
    assert metadata._match_generator("influx of photos") is None


def test_exif_camera(jpeg):
    r = metadata.inspect(jpeg({0x010F: "Apple", 0x0110: "iPhone 15 Pro"}))
    assert r["camera"] == "Apple iPhone 15 Pro"
    assert not r["declared_ai"]


def test_exif_software_generator(jpeg):
    assert metadata.inspect(jpeg({0x0131: "Midjourney v7"}))["declared_ai"]


def test_iptc_digital_source_type_in_xmp(jpeg):
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description '
           b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/'
           b'trainedAlgorithmicMedia"/></x:xmpmeta>')
    data = jpeg()
    # Splice an APP1 XMP segment right after SOI.
    hdr = b"http://ns.adobe.com/xap/1.0/\x00"
    seg = b"\xff\xe1" + (len(hdr) + len(xmp) + 2).to_bytes(2, "big") + hdr + xmp
    r = metadata.inspect(data[:2] + seg + data[2:])
    assert r["declared_ai"]
    assert any(f["key"] == "iptc_source" for f in r["findings"])
