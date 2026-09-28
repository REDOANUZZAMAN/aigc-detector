"""Fetch an image by URL without letting the URL reach anything internal.

Every hop (including each redirect) is resolved here, all resolved addresses
must be public, and the connection goes to the address we checked -- the
hostname only travels in the Host header and TLS SNI -- so a DNS answer that
changes between check and connect (rebinding) can't redirect us inward.
"""
import asyncio
import ipaddress
import socket
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit

import httpx

MAX_REDIRECTS = 3
TIMEOUT_S = 20
USER_AGENT = "AIGC-Detector/1.0 (+https://aigc.rmaikr.top)"


class FetchError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def _resolve(host: str, port: int) -> str:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise FetchError(f"Can't resolve {host}.")
    ips = [ipaddress.ip_address(i[4][0].split("%")[0]) for i in infos]
    if not ips or not all(_public(ip) for ip in ips):
        raise FetchError("That URL points to a private or reserved address.")
    return str(ips[0])


async def _get(url: str, max_bytes: int) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            u = urlsplit(url)
            if u.scheme not in ("http", "https") or not u.hostname:
                raise FetchError("Only http:// and https:// image URLs are supported.")
            port = u.port or (443 if u.scheme == "https" else 80)
            ip = await _resolve(u.hostname, port)
            target = u._replace(netloc=f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}")
            headers = {"Host": u.netloc.rsplit("@", 1)[-1], "User-Agent": USER_AGENT,
                       "Accept": "image/*"}
            ext = {"sni_hostname": u.hostname} if u.scheme == "https" else {}
            req = client.build_request("GET", target.geturl(), headers=headers, extensions=ext)
            r = await client.send(req, stream=True)
            try:
                if r.is_redirect and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code != 200:
                    raise FetchError(f"The server returned HTTP {r.status_code}.", 502)
                if int(r.headers.get("content-length") or 0) > max_bytes:
                    raise FetchError(f"Image is over {max_bytes // 1024 // 1024} MB.", 413)
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise FetchError(f"Image is over {max_bytes // 1024 // 1024} MB.", 413)
                name = PurePosixPath(unquote(u.path)).name or "image"
                return bytes(buf), name[:120]
            finally:
                await r.aclose()
    raise FetchError("Too many redirects.", 502)


async def fetch_image(url: str, max_bytes: int) -> tuple[bytes, str]:
    """Returns (bytes, filename). Raises FetchError with an HTTP status."""
    url = url.strip()
    if len(url) > 2048:
        raise FetchError("URL is too long.")
    try:
        return await asyncio.wait_for(_get(url, max_bytes), TIMEOUT_S)
    except asyncio.TimeoutError:
        raise FetchError("Timed out fetching the image.", 504)
    except httpx.HTTPError as e:
        raise FetchError(f"Couldn't fetch the image ({type(e).__name__}).", 502)
