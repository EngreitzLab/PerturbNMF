"""Disk-cached JSON over HTTP with retries, shared by every network step of the annotators.

A failed request is never cached, so a rerun retries exactly the requests that failed and
everything else is free.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


def read_body(response) -> bytes:
    """The whole body. A body cut short (seen behind some HTTPS proxies) is returned as far as it
    got, so the JSON parse fails and the caller retries rather than caching half a result."""
    chunks = []
    try:
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except http.client.IncompleteRead as exc:
        chunks.append(exc.partial)
    return b"".join(chunks)


class CachedHttp:
    """GET/POST JSON with retries, cached to disk by URL+body so reruns are free."""

    def __init__(self, cache_dir: Path, pause: float = 0.35):
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.cache_file = cache_dir / "http_cache.json"
        cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        # A failed request is never a result: drop it so the next run retries it.
        self.cache = {k: v for k, v in cache.items() if v is not None}
        self.dirty = 0

    def get_json(self, url: str, body: Optional[dict] = None, headers: Optional[dict] = None):
        key = url + ("|" + json.dumps(body, sort_keys=True) if body else "")
        if key in self.cache:
            return self.cache[key]
        data = json.dumps(body).encode() if body else None
        request = urllib.request.Request(
            url, data=data,
            headers={"Accept": "application/json", **({"Content-Type": "application/json"} if body else {}), **(headers or {})},
        )
        result = None
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.loads(response.read().decode() or "null")
                break
            except Exception as exc:
                if attempt == 5:
                    print(f"  giving up on {url[:100]} ({exc})")
                time.sleep(2 * attempt)
        time.sleep(self.pause)
        if result is None:
            return None
        self.cache[key] = result
        self.dirty += 1
        if self.dirty % 25 == 0:
            self.save()
        return result

    def post_form(self, url: str, fields: dict):
        """POST form fields, JSON back (the STRING API takes form fields, not a JSON body)."""
        body = urllib.parse.urlencode(fields, doseq=True).encode()
        key = url + "|form|" + body.decode()
        if key in self.cache:
            return self.cache[key]
        request = urllib.request.Request(url, data=body, headers={"Accept": "application/json"})
        result = None
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(read_body(response).decode() or "null")
                break
            except Exception as exc:
                if attempt == 5:
                    print(f"  giving up on {url[:100]} ({exc})")
                time.sleep(2 * attempt)
        time.sleep(self.pause)
        if result is None:
            return None
        self.cache[key] = result
        self.dirty += 1
        if self.dirty % 25 == 0:
            self.save()
        return result

    def save(self):
        self.cache_file.write_text(json.dumps(self.cache))
