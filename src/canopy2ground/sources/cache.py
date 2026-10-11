"""Response caches for fetched tiles and exports, keyed by string.

``SqliteCache`` keeps everything in one SQLite file per source, which stays
fast at the 10^5 to 10^6 small tiles a corpus build fetches (loose files that
many slow NTFS directory listing badly), doubles as the resume manifest, and
records definitive 404s so holes are not refetched. Transient errors are
never cached. ``MemoryCache`` is a bounded LRU with the same interface, for
one-off region fetches that should not leave a large cache on disk.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

OK, MISSING = "ok", "404"
Entry = tuple[str, bytes | None]          # (status, data); data is None for MISSING


class BlobCache(Protocol):
    def get(self, key: str) -> Entry | None: ...
    def put(self, key: str, status: str, data: bytes | None = None) -> None: ...


class SqliteCache:
    """Thread-safe: one connection per thread, writes serialised, WAL journal."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write = threading.Lock()
        with self._write, self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, status TEXT NOT NULL, "
                      "data BLOB, fetched_at REAL NOT NULL)")

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            c = sqlite3.connect(self.path, timeout=60)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return self._local.conn

    def get(self, key: str) -> Entry | None:
        row = self._conn().execute("SELECT status, data FROM blobs WHERE key = ?", (key,)).fetchone()
        return (row[0], row[1]) if row is not None and row[0] in (OK, MISSING) else None

    def put(self, key: str, status: str, data: bytes | None = None) -> None:
        if status not in (OK, MISSING):
            return
        with self._write, self._conn() as c:
            c.execute("INSERT INTO blobs (key, status, data, fetched_at) VALUES (?, ?, ?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET status = excluded.status, data = excluded.data, "
                      "fetched_at = excluded.fetched_at", (key, status, data, time.time()))


class MemoryCache:
    """Least-recently-used, at most ``capacity`` entries (0 caches nothing)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: OrderedDict[str, Entry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Entry | None:
        with self._lock:
            e = self._d.get(key)
            if e is not None:
                self._d.move_to_end(key)
            return e

    def put(self, key: str, status: str, data: bytes | None = None) -> None:
        if status not in (OK, MISSING) or self.capacity <= 0:
            return
        with self._lock:
            self._d[key] = (status, data)
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
