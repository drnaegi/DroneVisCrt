from __future__ import annotations

import io
import re
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RangeArchive(io.RawIOBase):
    def __init__(self, url: str, max_bytes: int = 512 * 1024**2):
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )))
        self.url = url
        self.position = 0
        self.transferred = 0
        self.max_bytes = max_bytes
        response = self.session.head(url, allow_redirects=True, timeout=(10, 30))
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.url = response.url
        self.etag = response.headers.get("ETag")
        self.last_modified = response.headers.get("Last-Modified")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in base or not 0 <= base[whence] + offset <= self.size:
            raise ValueError("Invalid ZIP seek")
        self.position = base[whence] + offset
        return self.position

    def read(self, size=-1):
        length = min(self.size - self.position, size if size >= 0 else self.size)
        if not length:
            return b""
        if self.transferred + length > self.max_bytes:
            raise ValueError("Range download budget exceeded")
        end = self.position + length - 1
        headers = {"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"}
        if self.etag and not self.etag.startswith("W/"):
            headers["If-Match"] = self.etag
        with self.session.get(self.url, headers=headers, timeout=(10, 45), stream=True) as response:
            response.raise_for_status()
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor the exact requested range")
            if self.etag and response.headers.get("ETag") != self.etag:
                raise ValueError("Remote archive changed")
            if self.last_modified and response.headers.get("Last-Modified") != self.last_modified:
                raise ValueError("Remote archive changed")
            content = response.raw.read(length + 1)
        self.transferred += len(content)
        if len(content) != length:
            raise ValueError("Truncated or oversized range response")
        self.position += length
        return content

    def close(self):
        self.session.close()
        super().close()


def safe_member(archive: zipfile.ZipFile, name: str, max_size: int = 64 * 1024**2) -> bytes:
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size > max_size:
        raise ValueError("Unexpected member size")
    if name.startswith(("/", "\\")) or ".." in re.split(r"[/\\]", name):
        raise ValueError("Unsafe archive path")
    return archive.read(info)
