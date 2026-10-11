import json
import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from canopy2ground.figures import panels, paper  # noqa: E402
from canopy2ground.mapping.warp import EAST_MAUI_3857  # noqa: E402

PX = (1080, 1350)                                     # 4.32 x 5.4 in at 250 dpi
RESULTS = Path(__file__).resolve().parents[1] / "results"


def _terrain(h, w, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 300 + 0.5 * yy + 30 * np.sin(xx / 17) * np.cos(yy / 23) + rng.normal(0, 0.5, (h, w))
    z[:, w // 3:w // 3 + 6] -= 40                     # a gorge
    return z.astype(np.float32)


@pytest.fixture
def panel_npz(tmp_path):
    h, w = 420, 400
    z = _terrain(h, w)
    sea = np.zeros((h, w), bool)
    sea[-30:] = True
    path = tmp_path / "panel.npz"
    naip = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    np.savez_compressed(path, naip=naip, sea=sea,
                        z_model=np.where(sea, 0, z), z_prior=np.where(sea, 0, z + 5).astype(np.float32),
                        transform=np.array([1.0, 0, 0, 0, -1.0, 0]))
    return path


@pytest.fixture
def overview_npz(tmp_path):
    h, w = 300, 240                                   # the East Maui 3857 grid at ~220 m/px
    yy, xx = np.mgrid[0:h, 0:w]
    sea = (xx - 120) ** 2 + (yy - 150) ** 2 > 110 ** 2
    lidar = ~sea & (xx < 60)
    path = tmp_path / "overview.npz"
    tr = EAST_MAUI_3857.transform
    bounds = (tr[2], tr[5] + tr[4] * EAST_MAUI_3857.height, tr[2] + tr[0] * EAST_MAUI_3857.width, tr[5])
    np.savez_compressed(path, shade=np.random.default_rng(2).integers(0, 255, (h, w), dtype=np.uint8),
                        lidar=lidar, sea=sea, bounds=np.array(bounds))
    return path


def _check_png(path):
    from PIL import Image

    with Image.open(path) as im:
        assert im.size == PX


def test_before_after(panel_npz, tmp_path):
    paper.before_after(panel_npz, tmp_path / "p1.png", "Test gulch")
    _check_png(tmp_path / "p1.png")


def test_full_figure(panel_npz, tmp_path):
    paper.full_figure(panel_npz, tmp_path / "p3.png", "Test gulch")
    _check_png(tmp_path / "p3.png")


def test_region_map(overview_npz, tmp_path):
    paper.region_map(overview_npz, tmp_path / "p2.png")
    _check_png(tmp_path / "p2.png")


def test_accuracy(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        paper.accuracy(tmp_path / "p4.png", RESULTS)
    _check_png(tmp_path / "p4.png")
    assert not caplog.records                         # the font resolved to an installed family


def test_accuracy_reads_the_published_numbers():
    groups, wins = paper.accuracy_groups(RESULTS)
    shown = [(n, [(m, f"{v:.2f}") for m, v in rows]) for _, n, _, rows in groups]
    assert shown == [
        ("n = 1,107 validation chips",
         [("Copernicus GLO-30", "1.27"), ("FABDEM", "1.25"), ("This model", "0.46")]),
        ("n = 2,643 ground segments, 28 overpasses", [("Copernicus GLO-30", "2.30"), ("This model", "1.26")]),
        ("n = 1,211 chips, region withheld from training",
         [("Copernicus GLO-30", "6.12"), ("FABDEM", "4.85"), ("Same method,\nretrained without it", "2.57")]),
    ]
    assert f"{wins:.0%}" == "99%"


def test_style_is_scoped(tmp_path):
    import matplotlib as mpl

    before = mpl.rcParams["font.size"]
    paper.accuracy(tmp_path / "p4.png", RESULTS)
    assert mpl.rcParams["font.size"] == before
    assert paper.FONT in paper.SANS


def test_copernicus_notice_is_verbatim():
    assert paper.COPERNICUS == (
        "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
        "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.")


def test_deepest_rows_finds_the_relief():
    hv = np.full((100, 50), 0.5)
    hv[60:70] = np.tile([0.1, 0.9], 25)                # the only textured band
    assert 50 <= paper.deepest_rows(hv, np.zeros_like(hv, bool), 20) <= 60


def test_multi_hillshade_flat_and_bounded():
    flat = panels.multi_hillshade(np.zeros((20, 20)))
    np.testing.assert_allclose(flat, np.sin(np.radians(40.0)))     # every sun at 40 degrees altitude
    rough = panels.multi_hillshade(_terrain(64, 64), spacing=2.0)
    assert rough.min() >= 0 and rough.max() <= 1 and rough.std() > 0.01


def test_crop_panel(tmp_path):
    from pyproj import Transformer

    from canopy2ground.grid import CHIP_M as CHIP
    from canopy2ground.grid import CRS

    _, lon, lat, size = panels.FIGURE_PLACE
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    stride, nx, ny = 256, 12, 12
    x0, y1 = float(np.floor(x)) - 1500, float(np.floor(y)) + 1500
    store = tmp_path / "store"
    store.mkdir()
    (store / "grid.json").write_text(json.dumps({"x0": x0, "y1": y1, "nx": nx, "ny": ny, "stride": stride,
                                                 "width": nx * stride + CHIP, "height": ny * stride + CHIP}))
    for iy in range(ny):
        naip = np.zeros((nx, CHIP, CHIP, 4), np.uint8)
        for ix in range(nx):
            naip[ix] = 10 * iy + ix
        np.savez(store / f"row_{iy:04d}.npz", naip=naip, naip_ok=np.ones(nx, bool))
    prior = tmp_path / "prior.tif"
    with rasterio.open(prior, "w", driver="GTiff", height=200, width=200, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(lon - 0.1, lat + 0.1, 0.001, 0.001)) as ds:
        ds.write(np.full((200, 200), 250.0, np.float32), 1)
    m = np.full((3000, 3000), 260.0, np.float32)
    m[:, :500] = np.nan                                                # the map's sea, west of the place
    with rasterio.open(tmp_path / "map.tif", "w", driver="GTiff", height=3000, width=3000, count=1,
                       dtype="float32", crs=CRS, transform=from_origin(x - 1500, y + 1500, 1.0, 1.0),
                       nodata=np.nan) as ds:
        ds.write(m, 1)
    out = tmp_path / "panel.npz"
    meta = panels.crop_panel(store, tmp_path / "map.tif", prior, out)
    z = np.load(out)
    assert z["naip"].shape[:2] == z["z_model"].shape == z["z_prior"].shape == tuple(meta["shape"])
    # Grid convergence (~1 degree here) makes the NE-minus-SW northing ~30 m longer than the side.
    assert abs(z["z_model"].shape[0] - size) < 40 and z["naip"].shape[2] == 3
    np.testing.assert_allclose(z["z_prior"][~z["sea"]], 250.0, atol=0.01)     # Lanczos, float32
    assert (z["z_model"][~z["sea"]] == 260.0).all() and not z["sea"].any()
    assert meta["naip_empty_frac"] == 0.0
    # The crop starts inside the right tile: its top-left pixel holds that tile's marker value.
    col, row = int((x - size / 2 * 1.0 - x0) // stride), int((y1 - (y + size / 2)) // stride)
    assert abs(int(z["naip"][0, 0, 0]) - (10 * min(row, ny - 1) + min(col, nx - 1))) <= 11


def test_overview(tmp_path):
    from pyproj import Transformer
    from shapely.geometry import box, mapping

    x0, y1, n = -17390000.0, 2380000.0, 700
    with rasterio.open(tmp_path / "surface.tif", "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs="EPSG:3857", transform=from_origin(x0, y1, 2.0, 2.0)) as ds:
        z = _terrain(n, n)
        z[:, -150:] = 0.0                                              # sea
        ds.write(z, 1)
    with rasterio.open(tmp_path / "model.tif", "w", driver="GTiff", height=1000, width=1000, count=1,
                       dtype="float32", crs="EPSG:3857",
                       transform=from_origin(x0 + 200, y1 - 200, 1.0, 1.0)) as ds:
        ds.write(np.zeros((1000, 1000), np.float32), 1)
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    w, s = to_ll(x0, y1 - 1400)
    e, nn = to_ll(x0 + 600, y1)
    (tmp_path / "fp.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(box(w, s, e, nn))}]}))
    out = tmp_path / "ov.npz"
    panels.overview(tmp_path / "surface.tif", tmp_path / "model.tif", tmp_path / "fp.geojson", out)
    ov = np.load(out)
    assert ov["shade"].shape == ov["lidar"].shape == ov["sea"].shape == (71, 71)      # 1000 m / 2 m / 7 px
    assert ov["shade"].dtype == np.uint8 and ov["sea"][:, -6:].all() and not ov["sea"][:, :50].any()
    assert ov["lidar"][:, :25].all() and not ov["lidar"][:, -20:].any()
    np.testing.assert_allclose(ov["bounds"], (x0 + 200, y1 - 1200, x0 + 1200, y1 - 200))
