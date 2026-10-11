import json
import sqlite3

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

from pyproj import Transformer  # noqa: E402
from rasterio.transform import Affine, from_origin  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

from canopy2ground.mapping import (  # noqa: E402
    coast_mask,
    footprint,
    lidar_mosaic,
    prior_tiles,
    surface,
    terrain_rgb,
)
from canopy2ground.mapping.warp import Grid, warp_to_3857  # noqa: E402

UTM = "EPSG:6634"
LON, LAT = -156.10, 20.80                        # inside the East Maui box


def _utm_origin():
    x, y = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(LON, LAT)
    return float(np.floor(x)), float(np.floor(y))


def _write(path, arr, transform, crs, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr, 1)


def _geojson(path, polys, crs=UTM, kinds=None):
    """Write projected shapely polygons as a lon/lat FeatureCollection."""
    from shapely.geometry import mapping
    from shapely.ops import transform

    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    feats = [{"type": "Feature", "properties": {"kind": k} if k else {},
              "geometry": mapping(transform(to_ll, p))} for p, k in zip(polys, kinds or [None] * len(polys))]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


# --- coast mask --------------------------------------------------------------------------------------------

def test_land_polygons_drops_isolated_specks_only():
    county = [box(0, 0, 100, 100)]
    speck_alone = box(200, 200, 205, 205)              # 25 m^2, touches nothing: whitewater
    speck_touching = box(98, 50, 103, 55)              # 25 m^2, overlaps County land: a rocky point
    big_alone = box(300, 300, 310, 310)                # 100 m^2 >= min_blob: kept regardless
    polys, kept, dropped = coast_mask.land_polygons(county, [speck_alone, speck_touching, big_alone], 50.0)
    assert (kept, dropped) == (2, 1)
    assert speck_alone not in polys and speck_touching in polys and big_alone in polys


def test_mask_block_sources():
    pred = np.array([[1.0, np.nan, 3.0, np.nan]], np.float32)
    land = np.array([[True, True, False, False]])
    covered = np.array([[True, True, True, False]])
    out, source, need = coast_mask.mask_block(pred, land, covered, np.nan)
    assert out[0, 0] == 1.0 and np.isnan(out[0, 2])                  # sea inside the coverage -> water
    assert np.isnan(out[0, 3]) and not need[0, 3]                    # outside the coverage: untouched
    assert need.tolist() == [[False, True, False, False]]            # land without a prediction -> prior
    assert source.tolist() == [[1, 0, 0, 0]]


def test_apply_coast_mask(tmp_path):
    x0, y0 = _utm_origin()
    n = 200
    pred = np.full((n, n), 50.0, np.float32)
    pred[:20, :20] = np.nan                                           # land the predictor skipped
    _write(tmp_path / "pred.tif", pred, from_origin(x0, y0 + n, 1.0, 1.0), UTM, nodata=np.nan)
    # County land: the left half. USACE adds a 10 x 10 m point on the coast and a 5 x 5 m speck offshore.
    _geojson(tmp_path / "county.geojson", [box(x0 - 50, y0 - 50, x0 + 100, y0 + n + 50)])
    _geojson(tmp_path / "usace.geojson",
             [box(x0 + 95, y0 + 100, x0 + 110, y0 + 110), box(x0 + 150, y0 + 150, x0 + 155, y0 + 155),
              box(x0 + 150, y0 + 20, x0 + 190, y0 + 60)],
             kinds=["land_above_mhw", "land_above_mhw", "valid_water_below_mhw"])
    prior = np.full((100, 100), 7.0, np.float32)
    _write(tmp_path / "prior.tif", prior, from_origin(LON - 0.05, LAT + 0.05, 0.001, 0.001), "EPSG:4326",
           nodata=-9999.0)
    clip = (LON - 0.02, LAT - 0.02, LON + 0.02, LAT + 0.02)
    out = tmp_path / "coast.tif"
    summary = coast_mask.apply_coast_mask(tmp_path / "pred.tif", out, tmp_path / "county.geojson",
                                          tmp_path / "usace.geojson", tmp_path / "prior.tif", clip_bbox=clip)
    with rasterio.open(out) as ds:
        z = ds.read(1)
    with rasterio.open(tmp_path / "coast_source.tif") as ds:
        src = ds.read(1)
    rows = n - 1 - np.arange(n)                                       # row index of northing y0 + k + 0.5
    assert np.all(z[:, :100][np.isfinite(z[:, :100])] > 0) and np.isfinite(z[20:, :100]).all()
    np.testing.assert_allclose(z[:20, :20], 7.0, atol=1e-3)          # filled from the prior
    assert (src[:20, :20] == 2).all() and (src[20:, :100] == 1).all()
    assert np.isfinite(z[rows[105], 105])                             # the USACE point is land
    assert np.isnan(z[rows[152], 152])                                # the offshore speck is dropped
    assert np.isnan(z[rows[40], 170])                                 # USACE water is not land
    assert np.isnan(z[:, 120:]).sum() > 0.95 * n * 80
    assert summary["copernicus_fill_px"] == 400
    assert summary["model_px"] == int((src == 1).sum())
    assert json.loads((tmp_path / "coast_summary.json").read_text())["min_blob_m2"] == 50.0


def test_apply_coast_mask_is_independent_of_the_block_size(tmp_path):
    x0, y0 = _utm_origin()
    n = 150
    yy, xx = np.mgrid[0:n, 0:n]
    pred = (20 + 0.1 * xx).astype(np.float32)
    pred[(xx - 40) ** 2 + (yy - 90) ** 2 < 15 ** 2] = np.nan       # land the predictor skipped
    _write(tmp_path / "pred.tif", pred, from_origin(x0, y0 + n, 1.0, 1.0), UTM, nodata=np.nan)
    _geojson(tmp_path / "county.geojson", [Point(x0 + 60, y0 + 70).buffer(55)])     # a round island
    _geojson(tmp_path / "usace.geojson", [box(x0 + 110, y0 + 10, x0 + 140, y0 + 40)],
             kinds=["land_above_mhw"])
    _write(tmp_path / "prior.tif", np.full((100, 100), 7.0, np.float32),
           from_origin(LON - 0.05, LAT + 0.05, 0.001, 0.001), "EPSG:4326")
    clip = (LON - 0.02, LAT - 0.02, LON + 0.02, LAT + 0.02)
    arrays = []
    for block in (n, 17):                                          # one window, then many with a ragged last
        out = tmp_path / f"coast_{block}.tif"
        coast_mask.apply_coast_mask(tmp_path / "pred.tif", out, tmp_path / "county.geojson",
                                    tmp_path / "usace.geojson", tmp_path / "prior.tif", clip_bbox=clip,
                                    block=block)
        with rasterio.open(out) as a, rasterio.open(tmp_path / f"coast_{block}_source.tif") as b:
            arrays.append((a.read(1), b.read(1)))
    (z1, s1), (z2, s2) = arrays
    np.testing.assert_array_equal(z1, z2)
    np.testing.assert_array_equal(s1, s2)
    assert {0, 1, 2} <= set(np.unique(s1).tolist())


# --- warp --------------------------------------------------------------------------------------------------

def test_warp_to_3857_bounds(tmp_path):
    from rasterio.warp import transform_bounds

    x0, y0 = _utm_origin()
    n = 300
    yy, xx = np.mgrid[0:n, 0:n]
    _write(tmp_path / "utm.tif", (100 + 0.1 * xx + 0.05 * yy).astype(np.float32),
           from_origin(x0, y0 + n, 1.0, 1.0), UTM, nodata=np.nan)
    with rasterio.open(tmp_path / "utm.tif") as s:
        w, s_, e, n_ = transform_bounds(s.crs, "EPSG:3857", *s.bounds)
    res = 2.0
    grid = Grid((res, 0.0, w - 100, 0.0, -res, n_ + 100), int((e - w + 200) / res),
                int((n_ - s_ + 200) / res))
    warp_to_3857(tmp_path / "utm.tif", tmp_path / "m.tif", grid, block=64)
    with rasterio.open(tmp_path / "m.tif") as ds:
        z, tr = ds.read(1), ds.transform
        assert ds.crs.to_epsg() == 3857 and ds.nodata is not None and np.isnan(ds.nodata)
    rows, cols = np.where(np.isfinite(z))
    left, top = tr * (cols.min(), rows.min())
    right, bottom = tr * (cols.max() + 1, rows.max() + 1)
    # The UTM square is slightly rotated in Mercator: the finite footprint fills its 3857 bounds to ~1%.
    tol = 0.02 * (e - w)
    assert abs(left - w) < tol and abs(right - e) < tol and abs(top - n_) < tol and abs(bottom - s_) < tol
    assert 99.9 < np.nanmin(z) and np.nanmax(z) < 100 + 0.15 * n + 0.1     # bilinear stays inside the range


# --- terrain-RGB -------------------------------------------------------------------------------------------

def test_terrain_rgb_round_trip():
    rng = np.random.default_rng(0)
    elev = np.concatenate([rng.uniform(-100, 4200, 5000), [0.0, 0.05, 0.1, 3055.3, -10000.0]]).reshape(5, -1)
    back = terrain_rgb.decode(terrain_rgb.encode(elev))
    err = elev - back
    assert err.min() > -1e-6 and err.max() < terrain_rgb.INTERVAL + 1e-6     # truncated to 0.1 m steps
    assert terrain_rgb.decode(terrain_rgb.encode(np.zeros((1, 1))))[0, 0] == pytest.approx(0.0)


def test_terrain_rgb_matches_rio_rgbify():
    enc = pytest.importorskip("rio_rgbify.encoders")
    elev = np.random.default_rng(1).uniform(-50, 4000, (64, 64))
    np.testing.assert_array_equal(terrain_rgb.encode(elev), enc.data_to_rgb(elev.copy(), -10000, 0.1))


def _mbtiles(path, tiles=((14, 1, 2, b"x"),)):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE metadata (name text, value text)")
    con.execute("CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, "
                "tile_data blob)")
    con.executemany("INSERT INTO metadata VALUES (?,?)",
                    [("format", "png"), ("format", "png"), ("name", "x")])
    con.executemany("INSERT INTO tiles VALUES (?,?,?,?)", tiles)
    con.commit()
    con.close()


def test_fix_metadata_replaces_duplicates(tmp_path):
    mb = tmp_path / "a.mbtiles"
    _mbtiles(mb)
    src = tmp_path / "s.tif"
    _write(src, np.zeros((10, 10), np.float32), from_origin(-17390000, 2380000, 10, 10), "EPSG:3857")
    terrain_rgb.fix_metadata(mb, 8, 14, "webp", src=src)
    rows = sqlite3.connect(mb).execute("SELECT name, value FROM metadata").fetchall()
    meta = dict(rows)
    assert len(rows) == len(meta)                                    # no duplicate keys left
    assert meta["format"] == "webp" and meta["minzoom"] == "8" and meta["maxzoom"] == "14"
    w, s, e, n = map(float, meta["bounds"].split(","))
    assert -156.3 < w < e < -156.1 and 20.8 < s < n < 20.95


def test_to_pmtiles(tmp_path):
    pytest.importorskip("pmtiles")
    from pmtiles.reader import MmapSource, Reader

    mb = tmp_path / "a.mbtiles"
    _mbtiles(mb, ((14, 1, 2, b"a"), (15, 3, 4, b"b")))
    src = tmp_path / "s.tif"
    _write(src, np.zeros((10, 10), np.float32), from_origin(-17390000, 2380000, 10, 10), "EPSG:3857")
    terrain_rgb.fix_metadata(mb, 14, 15, "webp", src=src)
    pm = tmp_path / "a.pmtiles"
    terrain_rgb.to_pmtiles(mb, pm, 14)
    with open(pm, "rb") as f:
        header = Reader(MmapSource(f)).header()
    assert header["max_zoom"] == 14 and header["addressed_tiles_count"] == 1


def test_tms_range():
    assert surface.tms_range(-1e-6, -1e-6, 1e-6, 1e-6, 0) == (0, 0, 0, 0)
    c0, c1, r0, r1 = surface.tms_range(0.001, 0.001, 0.002, 0.002, 10)          # just NE of (0, 0)
    assert (c0, c1) == (512, 512) and (r0, r1) == (512, 512)        # TMS rows count from the south


# --- surface merge -----------------------------------------------------------------------------------------

def test_merge_lidar_model_gap(tmp_path):
    """Lidar wins, the model is an exact 2x2 average, shallow water is 0, an enclosed void gets the prior."""
    pytest.importorskip("scipy")
    r = 1.0
    x0, y1 = -17390000.0, 2380000.0
    n = 512                                                          # output px at 2 m
    lidar = np.full((n, n), 100.0, np.float32)
    lidar[:, 2 * n // 3:] = np.nan                                   # surveyed: the left two thirds
    lidar[200:296, 100:196] = np.nan                                 # an enclosed 192 m hole
    _write(tmp_path / "lidar.tif", lidar, from_origin(x0, y1, 2 * r, 2 * r), "EPSG:3857", nodata=np.nan)
    m = 2 * n // 2                                                   # model: right half, 1 m px
    yy, xx = np.mgrid[0:2 * n, 0:m]
    model = (50 + 0.01 * xx + 0.02 * yy).astype(np.float32)
    model[-40:] = 0.2                                                # a strip of "seafloor"
    _write(tmp_path / "model.tif", model, from_origin(x0 + n * r, y1, r, r), "EPSG:3857", nodata=np.nan)
    _write(tmp_path / "prior.tif", np.full((n // 4, n // 4), 90.0, np.float32),
           from_origin(x0, y1, 8 * r, 8 * r), "EPSG:3857")
    out = tmp_path / "surface.tif"
    surface.merge(tmp_path / "model.tif", tmp_path / "lidar.tif", tmp_path / "prior.tif", out)
    with rasterio.open(out) as ds:
        z = ds.read(1)
        assert ds.transform.a == pytest.approx(2 * r) and z.shape == (n, n)
    avg = model.reshape(n, 2, m // 2, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(z[:-20, 2 * n // 3 + 2:], avg[:-20, n // 6 + 2:], atol=1e-4)   # model only
    assert (z[:, n // 2:2 * n // 3][np.isfinite(lidar[:, n // 2:2 * n // 3])] == 100.0).all()   # lidar wins
    assert (z[-20:, 2 * n // 3:] == 0.0).all()                                       # < 0.5 m -> sea
    np.testing.assert_allclose(z[210:286, 110:186], 100.0, atol=0.05)               # 90 + levelled +10


# --- prior, lidar mosaic, footprint ------------------------------------------------------------------------

def test_prior_3857(tmp_path):
    d = tmp_path / "cop"
    d.mkdir()
    z = np.full((120, 120), 500.0, np.float32)
    z[:, :30] = -32767.0                                             # voids
    z[:, 30:40] = -2.0                                               # below sea level
    _write(d / "tile.tif", z, from_origin(-156.2, 20.9, 0.001, 0.001), "EPSG:4326", nodata=-32767.0)
    out = prior_tiles.prior_3857(d, tmp_path / "prior_3857.tif", bbox=(-156.2, 20.78, -156.08, 20.9))
    with rasterio.open(out) as ds:
        p = ds.read(1)
        assert ds.crs.to_epsg() == 3857 and ds.transform.a == pytest.approx(prior_tiles.RES_3857)
    assert np.isfinite(p).all() and p.min() == 0.0                  # no NaN, nothing below 0
    assert np.isclose(p, 500.0, atol=1e-3).mean() > 0.5


def test_sources_in_box_and_mosaic(tmp_path):
    def tile(name, col, row, value):
        arr = np.full((8, 8), value, np.float32)
        tr = from_origin(lidar_mosaic.GRID_X0 + col * 0.5, lidar_mosaic.GRID_Y0 - row * 0.5, 0.5, 0.5)
        _write(tmp_path / name, arr, tr, UTM, nodata=-999999.0)

    x, y = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(-156.3, 20.8)
    col, row = int((x - lidar_mosaic.GRID_X0) / 0.5) // 4 * 4, int((lidar_mosaic.GRID_Y0 - y) / 0.5) // 4 * 4
    tile("a.tif", col, row, 10.0)
    tile("b.tif", col + 8, row, 20.0)
    src = "\n".join(f'<SimpleSource><SourceFilename relativeToVRT="0">{tmp_path / f}</SourceFilename>'
                    f'<DstRect xOff="{c}" yOff="{row}" xSize="8" ySize="8"/></SimpleSource>'
                    for f, c in (("a.tif", col), ("b.tif", col + 8)))
    far = '<SourceFilename>c.tif</SourceFilename><DstRect xOff="0" yOff="0" xSize="8" ySize="8"/>'
    srcs = lidar_mosaic.sources_in_box(src + far)
    assert [s[1:] for s in srcs] == [(col // 4, row // 4, 2, 2), (col // 4 + 2, row // 4, 2, 2)]
    assert lidar_mosaic.mosaic(srcs, tmp_path / "m.vrt", workers=2) == 0
    with rasterio.open(tmp_path / "m.vrt") as ds:
        np.testing.assert_array_equal(ds.read(1), [[10, 10, 20, 20], [10, 10, 20, 20]])
        assert ds.transform.a == 2.0 and ds.transform.c == lidar_mosaic.GRID_X0 + col * 0.5
    with pytest.raises(SystemExit):
        lidar_mosaic.sources_in_box('<SourceFilename>d.tif</SourceFilename><DstRect xOff="1" yOff="0" '
                                    'xSize="8" ySize="8"/>')


def test_coverage_polygons(tmp_path):
    n = 2000                                                         # 4 km at 2 m
    z = np.full((n, n), np.nan, np.float32)
    z[250:1750, 250:1750] = 10.0                                     # 3 x 3 km surveyed
    z[500:700, 500:700] = np.nan                                     # 0.16 km^2 pinhole: closed
    z[1000:1500, 1000:1500] = np.nan                                 # 1 km^2 void: kept
    x0, y0 = _utm_origin()
    _write(tmp_path / "mosaic.tif", z, from_origin(x0, y0, 2.0, 2.0), UTM, nodata=np.nan)
    parts, crs = footprint.coverage_polygons(tmp_path / "mosaic.tif")
    assert len(parts) == 1 and len(parts[0].interiors) == 1
    assert parts[0].area == pytest.approx(8.0e6, rel=1e-6)
    clip = tmp_path / "land.geojson"
    _geojson(clip, [box(x0, y0 - 4000, x0 + 2000, y0)])               # land: the western half only
    parts, _ = footprint.coverage_polygons(tmp_path / "mosaic.tif", clip)
    assert sum(p.area for p in parts) == pytest.approx(1500 * 3000, rel=1e-3)      # the pinhole stays closed


def test_affine_grid_constant():
    from canopy2ground.mapping.warp import EAST_MAUI_3857

    tr = Affine(*EAST_MAUI_3857.transform)
    right, bottom = tr * (EAST_MAUI_3857.width, EAST_MAUI_3857.height)
    assert right == pytest.approx(-17358173.206216607) and bottom == pytest.approx(2337589.8582207286)
