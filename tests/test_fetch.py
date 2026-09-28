import asyncio
import socket

import pytest

from app import fetch
from app.fetch import FetchError, fetch_image


def run(url):
    return asyncio.run(fetch_image(url, 1024))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/", "http://10.0.0.1/x.png",
    "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://0x7f000001/",
    "http://[::ffff:127.0.0.1]/",
])
def test_blocks_internal_addresses(url):
    with pytest.raises(FetchError, match="private or reserved"):
        run(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a.png", "not a url"])
def test_rejects_non_http(url):
    with pytest.raises(FetchError, match="Only http"):
        run(url)


def test_blocks_hostname_resolving_to_private(monkeypatch):
    async def fake_getaddrinfo(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", port))]
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(FetchError, match="private or reserved"):
        loop.run_until_complete(fetch_image("http://rebind.example/", 1024))
    loop.close()
