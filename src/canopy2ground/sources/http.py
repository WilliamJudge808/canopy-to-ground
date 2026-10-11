"""Polite HTTP client: one keep-alive session, a per-host concurrency cap, retries with backoff."""
from __future__ import annotations

import random
import threading
import time
from urllib.parse import urlparse

import requests

USER_AGENT = "canopy2ground/1.0 (+https://github.com/WilliamJudge808/canopy-to-ground)"


class Fetcher:
    """Thread-safe GET with at most ``per_host`` requests in flight to any one host.

    429 and 5xx responses and connection errors are retried with exponential
    backoff and jitter (``Retry-After`` is honoured); after ``retries``
    attempts the last response is returned, or the last exception raised.
    """

    def __init__(self, per_host: int = 8, retries: int = 5, backoff: float = 0.5, backoff_cap: float = 30.0,
                 timeout: float = 30.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        adapter = requests.adapters.HTTPAdapter(pool_connections=per_host, pool_maxsize=per_host,
                                                max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.per_host, self.retries, self.timeout = per_host, retries, timeout
        self.backoff, self.backoff_cap = backoff, backoff_cap
        self._hosts: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()

    def _slot(self, url: str) -> threading.BoundedSemaphore:
        host = urlparse(url).hostname or ""
        with self._lock:
            return self._hosts.setdefault(host, threading.BoundedSemaphore(self.per_host))

    def _delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return min(self.backoff_cap, self.backoff * 2 ** (attempt - 1)) + random.uniform(0, 0.25)

    def get(self, url: str) -> requests.Response:
        slot = self._slot(url)
        for attempt in range(1, self.retries + 2):
            try:
                with slot:
                    resp = self.session.get(url, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout):
                if attempt > self.retries:
                    raise
                time.sleep(self._delay(attempt))
                continue
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt <= self.retries:
                time.sleep(self._delay(attempt, resp.headers.get("Retry-After")))
                continue
            return resp
        raise AssertionError("unreachable")
