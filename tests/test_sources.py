import io
import sqlite3
import threading
import time

import numpy as np
import pytest

from canopy2ground import grid
from canopy2ground.sources import cache as cache_mod
from canopy2ground.sources import http
from canopy2ground.sources.cache import MemoryCache, SqliteCache

rasterio = pytest.importorskip("rasterio")
PIL = pytest.importorskip("PIL.Image")

BOUNDS = grid.chip_bounds("maui-+01544-+04489")


class Resp:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    @property
    def text(self):
        return self.content.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


class FakeFetcher:
    """Serves ``handler(url) -> Resp`` and records every URL."""

    def __init__(self, handler):
        self.handler, self.urls, self.lock = handler, [], threading.Lock()

    def get(self, url):
        with self.lock:
            self.urls.append(url)
        return self.handler(url)


def png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PIL.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# --- http -------------------------------------------------------------------------------------------------

def test_user_agent_names_the_project_and_carries_no_email():
    f = http.Fetcher()
    assert f.session.headers["User-Agent"] == http.USER_AGENT
    assert "canopy2ground" in http.USER_AGENT and "@" not in http.USER_AGENT


def test_gdal_env_sends_the_user_agent_only_inside_the_block():
    from rasterio._env import get_gdal_config

    from canopy2ground.sources.raster import gdal_env

    out = []
    with gdal_env():
        assert get_gdal_config("GDAL_HTTP_USERAGENT") == http.USER_AGENT
        t = threading.Thread(target=lambda: out.append(get_gdal_config("GDAL_DISABLE_READDIR_ON_OPEN")))
        t.start()
        t.join()
    assert out == ["EMPTY_DIR"]                         # worker threads see the main thread's options
    assert get_gdal_config("GDAL_HTTP_USERAGENT") is None


def test_fetcher_retries_server_errors_then_returns(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    f = http.Fetcher(retries=3)
    seq = iter([Resp(503), Resp(429, headers={"Retry-After": "0"}), Resp(200, b"ok")])
    monkeypatch.setattr(f.session, "get", lambda url, timeout: next(seq))
    assert f.get("https://example.org/a").content == b"ok"

    f = http.Fetcher(retries=2)
    calls = []
    monkeypatch.setattr(f.session, "get", lambda url, timeout: calls.append(url) or Resp(500))
    assert f.get("https://example.org/a").status_code == 500 and len(calls) == 3
    monkeypatch.setattr(f.session, "get", lambda url, timeout: Resp(404))
    assert f.get("https://example.org/a").status_code == 404           # not retried


def test_fetcher_caps_concurrency_per_host(monkeypatch):
    f = http.Fetcher(per_host=3)
    live, peak, lock = {}, {}, threading.Lock()

    def get(url, timeout):
        host = url.split("/")[2]
        with lock:
            live[host] = live.get(host, 0) + 1
            peak[host] = max(peak.get(host, 0), live[host])
        time.sleep(0.02)
        with lock:
            live[host] -= 1
        return Resp(200, b"x")

    monkeypatch.setattr(f.session, "get", get)
    threads = [threading.Thread(target=f.get, args=(f"https://h{i % 2}.org/{i}",)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == {"h0.org": 3, "h1.org": 3}


# --- caches -----------------------------------------------------------------------------------------------

def test_sqlite_cache_persists_hits_and_404s_but_not_errors(tmp_path):
    c = SqliteCache(tmp_path / "src" / "z19.sqlite")
    c.put("19/1/2", "ok", b"tile")
    c.put("19/1/3", "404")
    c.put("19/1/4", "error")
    again = SqliteCache(tmp_path / "src" / "z19.sqlite")
    assert again.get("19/1/2") == ("ok", b"tile")
    assert again.get("19/1/3") == ("404", None)
    assert again.get("19/1/4") is None and again.get("nope") is None
    out = []
    t = threading.Thread(target=lambda: out.append(again.get("19/1/2")))
    t.start()
    t.join()
    assert out == [("ok", b"tile")]


def test_sqlite_cache_reads_and_extends_an_older_wider_schema(tmp_path):
    p = tmp_path / "old.sqlite"
    with sqlite3.connect(p) as db:
        db.execute("CREATE TABLE blobs (key TEXT PRIMARY KEY, status TEXT NOT NULL, content_type TEXT, "
                   "data BLOB, byte_size INTEGER NOT NULL DEFAULT 0, fetched_at REAL NOT NULL, "
                   "attempts INTEGER NOT NULL DEFAULT 1, error TEXT)")
        db.execute("INSERT INTO blobs VALUES ('a', 'ok', 'image/jpeg', x'ff', 1, 0, 1, NULL)")
        db.execute("INSERT INTO blobs VALUES ('b', 'error', NULL, NULL, 0, 0, 1, 'HTTP 500')")
    c = SqliteCache(p)
    assert c.get("a") == ("ok", b"\xff")
    assert c.get("b") is None                          # a recorded error means "fetch again"
    c.put("b", "ok", b"new")
    assert c.get("b") == ("ok", b"new")


def test_memory_cache_is_lru_bounded():
    c = MemoryCache(2)
    c.put("a", "ok", b"1")
    c.put("b", "ok", b"2")
    c.get("a")
    c.put("c", "404")
    assert c.get("b") is None and c.get("a") == ("ok", b"1") and c.get("c") == ("404", None)
    none = MemoryCache(0)
    none.put("a", "ok", b"1")
    assert none.get("a") is None
    assert cache_mod.OK == "ok"


# --- pictometry -------------------------------------------------------------------------------------------

def test_tile_math_is_self_consistent():
    from canopy2ground.sources import pictometry as p

    for x, y in [(34158, 231042), (31000, 229000), (0, 0)]:
        assert p.tiles_covering(*p.tile_bounds(x, y)) == [(x, y)]
    xmin, ymin, xmax, ymax = p.tile_bounds(10, 20)
    assert xmax - xmin == pytest.approx(p.tile_size()) and ymax - ymin == pytest.approx(p.tile_size())
    assert p.tile_bounds(10, 21)[3] == pytest.approx(ymin)          # y counts southward


def test_mosaic_places_tiles_and_replicates_grey():
    from canopy2ground.sources import pictometry as p

    tiles = {(5, 7): np.full((3, 256, 256), 10, np.uint8), (6, 8): np.full((1, 256, 256), 200, np.uint8)}
    arr, transform = p.mosaic(tiles, zoom=4)
    assert arr.shape == (3, 512, 512)
    assert (arr[:, :256, :256] == 10).all() and (arr[:, 256:, 256:] == 200).all()
    assert (arr[:, :256, 256:] == 0).all()                           # absent tile stays empty
    xmin, _, _, ymax = p.tile_bounds(5, 7, zoom=4)
    assert (transform.c, transform.f) == pytest.approx((xmin, ymax))
    assert transform.a == pytest.approx(p.tile_size(4) / 256)


def test_pictometry_read_fetches_row_before_column_and_reports_tile_coverage(tmp_path):
    from rasterio.warp import transform_bounds

    from canopy2ground.sources import pictometry as p

    need = p.tiles_covering(*transform_bounds(grid.CRS, "EPSG:3857", *BOUNDS, densify_pts=21))
    holes = set(need[:5])

    def handler(url):
        z, y, x = (int(v) for v in url.rsplit("/", 3)[1:])
        assert z == 19
        return Resp(404) if (x, y) in holes else Resp(200, png(np.full((256, 256, 3), 120, np.uint8)))

    f, cache = FakeFetcher(handler), SqliteCache(tmp_path / "t.sqlite")
    rgb, coverage = p.read(BOUNDS, grid.CRS, f, cache, workers=4)
    assert rgb.shape == (3, 512, 512) and rgb.dtype == np.uint8
    assert coverage == pytest.approx(1 - len(holes) / len(need))
    assert {tuple(int(v) for v in u.rsplit("/", 2)[1:])[::-1] for u in f.urls} == set(need)
    assert np.median(rgb) == 120
    n = len(f.urls)
    p.read(BOUNDS, grid.CRS, f, cache)                               # hits and 404s both come from the cache
    assert len(f.urls) == n

    empty, cov = p.read(BOUNDS, grid.CRS, FakeFetcher(lambda u: Resp(404)))
    assert cov == 0 and empty.shape == (3, 512, 512) and not empty.any()


# --- vivid ------------------------------------------------------------------------------------------------

def test_vivid_export_request_keeps_the_cache_key_scheme():
    import hashlib

    from canopy2ground.sources import vivid

    url, key, (xmin, ymin, xmax, ymax) = vivid.export_request(BOUNDS, grid.CRS)
    w, h = round((xmax - xmin) / 0.6), round((ymax - ymin) / 0.6)
    raw = f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f},({w}, {h})"
    assert key == "vivid_" + hashlib.sha256(raw.encode()).hexdigest()[:24]
    assert f"size={w},{h}" in url and "bboxSR=3857" in url and "format=jpg" in url


def test_vivid_read_warps_and_measures_non_black_coverage():
    from canopy2ground.sources import vivid

    def handler(url):
        w, h = (int(v) for v in url.split("size=")[1].split("&")[0].split(","))
        img = np.full((h, w, 3), 90, np.uint8)
        img[:, : w // 2] = 0
        return Resp(200, png(img))

    rgb, coverage = vivid.read(BOUNDS, grid.CRS, FakeFetcher(handler))
    assert rgb.shape == (3, 512, 512) and 0.4 < coverage < 0.6
    assert (rgb[:, :, -10:] == 90).all() and (rgb[:, :, :10] == 0).all()
    with pytest.raises(OSError):
        vivid.read(BOUNDS, grid.CRS, FakeFetcher(lambda u: Resp(500)))


# --- lidar / naip -----------------------------------------------------------------------------------------

def write_tif(path, arr, crs, x0, y1, res, nodata):
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype, crs=crs, transform=from_origin(x0, y1, res, res), nodata=nodata) as d:
        d.write(arr)


def test_localize_vrt_makes_relative_sources_absolute():
    from canopy2ground.sources import lidar

    vrt = ('<SourceFilename relativeToVRT="1">blk_1/4QGJ365385.tif</SourceFilename>'
           '<SourceFilename relativeToVRT="0">/abs/x.tif</SourceFilename>')
    out = lidar.localize_vrt(vrt, base="https://h/dem")
    assert ('<SourceFilename relativeToVRT="0">/vsicurl/https://h/dem/blk_1/4QGJ365385.tif</SourceFilename>'
            in out)
    assert '<SourceFilename relativeToVRT="0">/abs/x.tif</SourceFilename>' in out


def test_fetch_vrt_writes_a_localized_copy(tmp_path):
    from canopy2ground.sources import lidar

    vrt = '<SourceFilename relativeToVRT="1">blk_1/a.tif</SourceFilename>'
    f = FakeFetcher(lambda url: Resp(200, vrt.encode()))
    out = lidar.fetch_vrt(tmp_path / "data" / "lidar.vrt", f)
    assert f.urls == [lidar.VRT_URL]
    assert out.read_text(encoding="utf-8") == lidar.localize_vrt(vrt)
    with pytest.raises(OSError):
        lidar.fetch_vrt(tmp_path / "x.vrt", FakeFetcher(lambda url: Resp(503)))


def test_lidar_read_marks_voids_and_measures_coverage(tmp_path):
    from canopy2ground.sources import lidar

    dem = np.full((1, 1024, 1024), 250.0, np.float32)
    dem[:, :, :512] = lidar.NODATA                                    # west half void (0.5 m pixels)
    write_tif(tmp_path / "dtm.tif", dem, grid.CRS, BOUNDS.xmin, BOUNDS.ymax, 0.5, lidar.NODATA)
    elev, coverage = lidar.read(BOUNDS, grid.CRS, str(tmp_path / "dtm.tif"))
    assert elev.shape == (512, 512) and coverage == pytest.approx(0.5, abs=0.01)
    assert (elev[:, 300:] == 250).all() and (elev[:, :200] == lidar.NODATA).all()


def test_naip_read_four_bands(tmp_path):
    from canopy2ground.sources import naip

    img = np.stack([np.full((600, 600), v, np.uint8) for v in (10, 20, 30, 40)])
    # NAIP ships in NAD83 / UTM 4N; the chip grid is NAD83(PA11) / UTM 4N, sub-metre apart
    write_tif(tmp_path / "naip.tif", img, "EPSG:26904", BOUNDS.xmin - 20, BOUNDS.ymax + 20, 1.0, 0)
    arr, coverage = naip.read(BOUNDS, grid.CRS, str(tmp_path / "naip.tif"))
    assert arr.shape == (4, 512, 512) and arr.dtype == np.float32 and coverage == 1.0
    assert np.allclose(arr[:, 256, 256], [10, 20, 30, 40])


def test_read_layer_returns_hwc_uint8_and_zeros_when_uncovered(monkeypatch):
    from canopy2ground import sources

    monkeypatch.setattr(sources.naip, "read", lambda b, c: (np.full((4, 512, 512), 300.7, np.float32), 0.8))
    img, cov = sources.read_layer("naip", BOUNDS, grid.CRS, None)
    assert img.shape == (512, 512, 4) and img.dtype == np.uint8 and (img == 255).all() and cov == 0.8
    monkeypatch.setattr(sources.naip, "read", lambda b, c: (np.full((4, 512, 512), 9.0, np.float32), 0.0))
    img, cov = sources.read_layer("naip", BOUNDS, grid.CRS, None)
    assert cov == 0 and not img.any()
    with pytest.raises(ValueError):
        sources.read_layer("h1977", BOUNDS, grid.CRS, None)
