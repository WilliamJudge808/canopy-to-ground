import json

import numpy as np
import pytest

from canopy2ground import grid, prefetch

rasterio = pytest.importorskip("rasterio")

S = 512


def test_region_grid_covers_the_box_with_overlapping_tiles():
    g = prefetch.region_grid(prefetch.PRESETS["keanae"], overlap=64)
    assert g["stride"] == 448 and g["chip"] == S and g["crs"] == grid.CRS
    assert g["layers"] == ["vivid", "naip"]
    # the last tile reaches the far edge, one fewer would not
    for n, size in ((g["nx"], g["width"]), (g["ny"], g["height"])):
        assert (n - 1) * g["stride"] + S >= size > (n - 2) * g["stride"] + S
    assert 6000 < g["width"] < 6500 and 5400 < g["height"] < 5700
    x0, y0, x1, y1 = prefetch.tile_bounds(g, 1, 2)
    assert (x0, y1) == (g["x0"] + 448, g["y1"] - 896) and (x1 - x0, y1 - y0) == (S, S)


@pytest.fixture
def region(tmp_path, monkeypatch):
    """A 2 x 2 tile region whose prior is void (sea) over the west half of the left column."""
    from rasterio.transform import from_origin

    g = {"x0": 744000.0, "y1": 2300000.0, "nx": 2, "ny": 2, "stride": 448, "chip": S, "width": 960,
         "height": 960, "layers": ["vivid", "naip"], "crs": grid.CRS}
    dem = np.full((1, 1000, 1000), 300.0, np.float32)
    dem[:, :, :200] = np.nan
    with rasterio.open(tmp_path / "prior.tif", "w", driver="GTiff", height=1000, width=1000, count=1,
                       dtype="float32", crs=grid.CRS, transform=from_origin(g["x0"], g["y1"], 1.0, 1.0),
                       nodata=np.nan) as d:
        d.write(dem)

    def pictometry(bounds, crs, fetcher, cache, workers):
        right = bounds[0] > g["x0"]
        return np.full((3, S, S), 50, np.uint8), (1.0 if right else 0.5)    # left column falls back to Vivid

    def vivid(bounds, crs, fetcher, cache=None):
        return np.full((3, S, S), 90, np.uint8), 1.0

    def layer(name, bounds, crs, fetcher, cache=None):
        if name == "naip":
            raise OSError("no NAIP")
        return np.full((S, S, 3), 7, np.uint8), (0.6 if bounds[3] == g["y1"] else 0.4)

    monkeypatch.setattr(prefetch.pictometry, "read", pictometry)
    monkeypatch.setattr(prefetch.vivid, "read", vivid)
    monkeypatch.setattr(prefetch, "read_layer", layer)
    return g, tmp_path / "prior.tif"


def test_prefetch_rows_match_what_predict_reads(region, tmp_path):
    g, prior = region
    store = tmp_path / "store"
    assert prefetch.prefetch(store, g, prior, workers=2) == 2
    saved = json.loads((store / "grid.json").read_text())
    assert saved == g and {"width", "height", "x0", "y1", "nx", "ny", "stride"} <= set(saved)
    rows = []
    for iy in range(2):
        with np.load(store / f"row_{iy:04d}.npz") as z:
            rows.append({k: z[k] for k in z.files})
    for row in rows:
        assert set(row) == {"skip", "rgb", "rgb_ok", "vivid", "vivid_ok", "naip", "naip_ok"}
        assert row["rgb"].shape == (2, S, S, 3) and row["naip"].shape == (2, S, S, 4)
        # left tile: prior 312 / 512 valid, so it is fetched
        assert not row["skip"].any() and row["rgb_ok"].all()
        assert (row["rgb"][0] == 90).all() and (row["rgb"][1] == 50).all()      # Vivid fallback at <= 0.5
        assert not row["naip_ok"].any() and not row["naip"].any()
    assert rows[0]["vivid_ok"].all() and not rows[1]["vivid_ok"].any()          # layer needs coverage >= 0.5
    assert not rows[1]["vivid"].any()


def test_prefetch_skips_sea_tiles_and_resumes(region, tmp_path, monkeypatch):
    g, prior = region
    g = dict(g, x0=g["x0"] - 200)                          # left column now under half valid
    store = tmp_path / "store"
    assert prefetch.prefetch(store, g, prior, max_rows=1) == 1
    with np.load(store / "row_0000.npz") as z:
        assert z["skip"].tolist() == [True, False] and z["rgb_ok"].tolist() == [False, True]
        assert not z["rgb"][0].any()
    before = (store / "row_0000.npz").stat().st_mtime_ns
    assert prefetch.prefetch(store, g, prior) == 2
    assert (store / "row_0000.npz").stat().st_mtime_ns == before
    with pytest.raises(SystemExit):
        prefetch.prefetch(store, dict(g, stride=400), prior)


def test_cli_maps_a_model_with_no_extra_layers(monkeypatch, tmp_path):
    from canopy2ground.mapping import EAST_MAUI_BBOX

    seen = {}
    monkeypatch.setattr(prefetch, "prefetch", lambda store, g, *a: seen.update(g))
    prefetch.main(["--preset", "keanae", "--prior", "p.tif", "--out", str(tmp_path), "--layers"])
    assert seen["layers"] == []
    prefetch.main(["--preset", "keanae", "--prior", "p.tif", "--out", str(tmp_path)])
    assert seen["layers"] == ["vivid", "naip"]
    assert prefetch.PRESETS["east_maui"] == EAST_MAUI_BBOX
