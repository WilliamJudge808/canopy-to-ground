import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from canopy2ground import grid
from canopy2ground.build import arrays, chips, regions
from canopy2ground.build.cache import NpyWriter, assemble, chip_ids, row_order

rasterio = pytest.importorskip("rasterio")

S = 512


def test_npy_writer_matches_np_save(tmp_path):
    a = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    with NpyWriter(tmp_path / "a.npy", a.shape, np.float32) as w:
        for row in a.astype(np.float64):                   # rows are cast to the declared dtype
            w.append(row)
    np.save(tmp_path / "ref.npy", a)
    assert (tmp_path / "a.npy").read_bytes() == (tmp_path / "ref.npy").read_bytes()
    assert np.array_equal(np.load(tmp_path / "a.npy", mmap_mode="r"), a)


def test_npy_writer_publishes_nothing_when_incomplete(tmp_path):
    with pytest.raises(ValueError):
        with NpyWriter(tmp_path / "b.npy", (2, 3), np.uint8) as w:
            w.append(np.zeros(3))
    with pytest.raises(RuntimeError):
        with NpyWriter(tmp_path / "c.npy", (2, 3), np.uint8) as w:
            w.append(np.zeros(3))
            raise RuntimeError
    assert not list(tmp_path.iterdir())


def test_row_order_is_sorted_then_seeded_shuffle_then_buffer_drop():
    ids = [grid.chip_id(1540, 4470 + k) for k in range(10)]          # 4479 is in the val buffer
    expect = sorted(ids)
    np.random.default_rng(1234).shuffle(expect)
    order, n_buf = row_order(list(reversed(ids)), "val")
    assert order == expect and n_buf == 0
    order, n_buf = row_order(ids, "train")
    assert order == [c for c in expect if c != grid.chip_id(1540, 4479)] and n_buf == 1


def corpus_chip(seed: int, void_cols: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    lidar = rng.normal(300, 20, (S, S)).astype(np.float32)
    m = np.ones((S, S), bool)
    m[:, :void_cols] = False
    lidar[~m] = 0.0
    return {"rgb": rng.integers(0, 255, (S, S, 3), dtype=np.uint8), "lidar": lidar, "lmask": m}


def make_corpus(root, cells):
    for k, (ix, iy) in enumerate(cells):
        d = root / grid.chip_split(ix, iy)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{grid.chip_id(ix, iy)}.npz", **corpus_chip(k, void_cols=50 * k))


def test_assemble_writes_the_dataset_layout(tmp_path):
    cells = [(1540, 4475), (1540, 4476), (1541, 4476), (1540, 4479), (1544, 4489), (1545, 4489)]
    make_corpus(tmp_path / "corpus", cells)
    assemble(tmp_path / "corpus", tmp_path / "cache")
    for split, n in (("train", 3), ("val", 2)):
        d = tmp_path / "cache" / split
        meta = json.loads((d / "meta.json").read_text())
        assert meta["n"] == n and [c["i"] for c in meta["chips"]] == list(range(n))
        corpus_ids = [p.stem for p in (tmp_path / "corpus" / split).glob("*.npz")]
        assert chip_ids(d) == row_order(corpus_ids, split)[0]
        rgb, lid, lm = (np.load(d / f, mmap_mode="r") for f in ("rgb.npy", "lidar.npy", "lmask.npy"))
        assert rgb.shape == (n, S, S, 3) and rgb.dtype == np.uint8
        assert lid.dtype == np.float32 and lm.dtype == np.bool_
        for row, c in enumerate(meta["chips"]):
            with np.load(tmp_path / "corpus" / split / f"{c['id']}.npz") as z:
                assert np.array_equal(rgb[row], z["rgb"]) and np.array_equal(lid[row], z["lidar"])
                v = z["lidar"][z["lmask"]]
            assert c["relief"] == pytest.approx(float(v.max() - v.min()))
            assert c["lidar_cov"] == pytest.approx(lm[row].mean())
    assert json.loads((tmp_path / "cache" / "train" / "meta.json").read_text())["n_buffer_excluded"] == 1


def test_chip_ids_accept_the_older_path_form(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"n": 2, "chips": [
        {"i": 0, "path": "D:\\corpus\\val\\maui-+01544-+04489.npz"}, {"i": 1, "id": "maui-+01545-+04489"}]}))
    assert chip_ids(tmp_path) == ["maui-+01544-+04489", "maui-+01545-+04489"]


# --- chip selection ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("coverage,rgb_coverage,kept",
                         [(0.39, 1.0, False), (0.40, 1.0, True), (1.0, 0.0, False), (1.0, 0.01, True)])
def test_fetch_chip_selection_rule(monkeypatch, coverage, rgb_coverage, kept):
    elev = np.full((S, S), 100.0, np.float32)
    n_void = int(round((1 - coverage) * S * S))
    elev.flat[:n_void] = -999999.0
    monkeypatch.setattr(chips.lidar, "read", lambda b, c, v: (elev.copy(), coverage))
    monkeypatch.setattr(chips.pictometry, "read",
                        lambda b, c, f, cache, workers: (np.full((3, S, S), 7, np.uint8), rgb_coverage))
    out = chips.fetch_chip("maui-+01544-+04489", "x.vrt", None)
    assert (out is not None) == kept
    if kept:
        assert out["rgb"].shape == (S, S, 3) and (out["rgb"] == 7).all()
        assert out["lmask"].sum() == S * S - n_void and (out["lidar"][~out["lmask"]] == 0).all()


def test_fetch_chip_needs_1024_label_pixels(monkeypatch):
    elev = np.full((S, S), -999999.0, np.float32)
    elev.flat[:1023] = 5.0
    monkeypatch.setattr(chips.lidar, "read", lambda b, c, v: (elev, 1.0))   # coverage passes, mask fails
    assert chips.fetch_chip("maui-+01544-+04489", "x.vrt", None) is None


def test_fetch_corpus_writes_splits_and_resumes(monkeypatch, tmp_path):
    cells = [(1544, 4489), (1540, 4476), (1540, 4477)]
    monkeypatch.setattr(chips.grid, "chips_in_lonlat", lambda *box: cells)
    calls = []

    def fake(cid, vrt, fetcher, cache, workers=2):
        calls.append(cid)
        return None if cid.endswith("4477") else corpus_chip(0)

    monkeypatch.setattr(chips, "fetch_chip", fake)
    chips.fetch_corpus(tmp_path, "x.vrt")
    assert sorted(p.name for p in (tmp_path / "val").iterdir()) == ["maui-+01544-+04489.npz"]
    assert sorted(p.name for p in (tmp_path / "train").iterdir()) == ["maui-+01540-+04476.npz"]
    assert (tmp_path / "rejected.txt").read_text().split() == ["maui-+01540-+04477"]
    calls.clear()
    chips.fetch_corpus(tmp_path, "x.vrt")
    assert calls == []


# --- per-chip arrays --------------------------------------------------------------------------------------

def tiny_cache(root, ids_by_split):
    for split, ids in ids_by_split.items():
        d = root / split
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"n": len(ids), "chips": [{"i": i, "id": c} for i, c in
                                                                           enumerate(ids)]}))
    return root


def test_build_layers_keeps_row_order_and_zero_fills_failures(monkeypatch, tmp_path):
    ids = [grid.chip_id(1540 + k, 4470) for k in range(12)]
    cache = tiny_cache(tmp_path / "cache", {"val": ids})

    def fake(name, bounds, crs, fetcher, cache=None):
        time.sleep(random.random() / 200)
        k = int(bounds[0] // 512) - 1540
        if name == "naip" and k == 3:
            raise OSError("server down")
        c = 3 if name == "vivid" else 4
        return np.full((S, S, c), k, np.uint8), k / 10

    monkeypatch.setattr(arrays, "read_layer", fake)
    arrays.build_layers(cache, tmp_path / "layers", splits=("val",), workers=4)
    d = tmp_path / "layers" / "val"
    vivid, naip = np.load(d / "img_vivid.npy"), np.load(d / "img_naip.npy")
    assert vivid.shape == (12, S, S, 3) and naip.shape == (12, S, S, 4)
    assert [int(vivid[k, 0, 0, 0]) for k in range(12)] == list(range(12))
    assert not naip[3].any() and np.load(d / "avail_naip.npy")[3] == 0
    assert np.allclose(np.load(d / "avail_vivid.npy"), np.arange(12) / 10)
    assert threading.active_count() < 8


def test_build_land_rasterises_the_coastline(tmp_path):
    pytest.importorskip("shapely")
    from pyproj import Transformer

    from canopy2ground.coast import Coastline

    land_id, sea_id = "maui-+01544-+04489", "maui-+01600-+04489"
    b = grid.chip_bounds(land_id)
    to_ll = Transformer.from_crs(grid.CRS, "EPSG:4326", always_xy=True).transform
    ring = [to_ll(x, y) for x, y in [(b.xmin - 100, b.ymin - 100), (b.xmin + 200, b.ymin - 100),
                                     (b.xmin + 300, b.ymax + 100), (b.xmin - 100, b.ymax + 100)]]
    poly = {"type": "Polygon", "coordinates": [ring + ring[:1]]}
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": poly}]}
    (tmp_path / "coast.geojson").write_text(json.dumps(gj))
    cache = tiny_cache(tmp_path / "cache", {"val": [land_id, sea_id]})
    arrays.build_land(cache, tmp_path / "coast.geojson", splits=("val",))
    land = np.load(cache / "val" / "land.npy")
    assert land.dtype == np.uint8 and land.shape == (2, S, S)
    assert np.array_equal(land[0], Coastline(tmp_path / "coast.geojson", grid.CRS).land_mask(b, S))
    assert 0.3 < land[0].mean() < 0.6 and land[0][:, :150].all() and not land[0][:, 350:].any()
    assert not land[1].any()


def test_build_prior_warps_and_masks(tmp_path):
    from rasterio.transform import from_origin

    dem = np.full((40, 40), 500.0, np.float32)
    dem[:, :20] = np.nan                                    # void west of -156.10
    with rasterio.open(tmp_path / "dem.tif", "w", driver="GTiff", height=40, width=40, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(-156.12, 20.86, 0.001, 0.001),
                       nodata=np.nan) as d:
        d.write(dem, 1)
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", grid.CRS, always_xy=True).transform(-156.10, 20.84)
    ix, iy = grid.chip_index(x, y)
    inside = grid.chip_id(ix + 2, iy)                       # entirely east of the void
    edge = grid.chip_id(ix, iy)                             # straddles it
    cache = tiny_cache(tmp_path / "cache", {"val": [inside, edge]})
    arrays.build_prior(cache, tmp_path / "dem.tif", splits=("val",))
    e, m = np.load(cache / "val" / "dem_copernicus.npy"), np.load(cache / "val" / "dmask_copernicus.npy")
    assert e.dtype == np.float32 and m.dtype == np.bool_
    assert m[0].all() and np.allclose(e[0], 500, atol=1e-3)
    assert 0 < m[1].mean() < 1 and (e[1][~m[1]] == 0).all()


# --- regions ----------------------------------------------------------------------------------------------

def test_regions_name_components_by_nearest_centroid_and_split_molokai(tmp_path):
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", grid.CRS, always_xy=True).transform

    def blob(lon, lat, n=3):
        ix, iy = grid.chip_index(*to_utm(lon, lat))
        return [(ix + dx, iy + dy) for dx in range(n) for dy in range(n)]

    lanai = blob(-156.93, 20.82)
    (ix0, iy), (ix1, _) = grid.chip_index(*to_utm(-157.10, 21.10)), grid.chip_index(*to_utm(-156.90, 21.10))
    molokai = [(ix, iy) for ix in range(ix0, ix1)]                 # one connected row of chips across the cut
    cells = lanai + molokai
    labels = regions.label_cells(set(cells))
    assert {labels[c] for c in lanai} == {"lanai"}
    to_ll = Transformer.from_crs(grid.CRS, "EPSG:4326", always_xy=True).transform
    for c in molokai:
        lon = to_ll((c[0] + 0.5) * 512, (c[1] + 0.5) * 512)[0]
        want = "molokai_wet" if lon > -157.0 else "molokai_buffer" if lon > -157.02 else "molokai_dry"
        assert labels[c] == want
    assert {"molokai_wet", "molokai_buffer", "molokai_dry"} <= set(labels.values())

    ids = [grid.chip_id(*c) for c in cells]
    cache = tiny_cache(tmp_path / "cache", {"train": ids[::2], "val": ids[1::2]})
    out = regions.build_regions(cache)
    assert json.loads((cache / "island_labels.json").read_text()) == out
    assert out["val"]["0"] == labels[cells[1]]


def test_components_are_8_connected():
    comps = regions.components({(0, 0), (1, 1), (5, 5), (5, 6)})
    assert sorted(sorted(c) for c in comps) == [[(0, 0), (1, 1)], [(5, 5), (5, 6)]]


@pytest.mark.data
def test_real_cache_obeys_the_split_and_reproduces_its_region_labels():
    if not os.environ.get("C2G_CACHE"):
        pytest.skip("set C2G_CACHE to the training cache root")
    root = Path(os.environ["C2G_CACHE"])
    ids = {s: chip_ids(root / s) for s in ("train", "val")}
    for split, v in ids.items():
        cells = [grid.parse_chip_id(c) for c in v]
        assert all(grid.chip_split(*c) == split and not grid.in_buffer(*c) for c in cells)
    assert row_order(ids["val"], "val")[0] == ids["val"]
    want = json.loads((root / "island_labels.json").read_text())
    cells = {s: [grid.parse_chip_id(c) for c in v] for s, v in ids.items()}
    labels = regions.label_cells({c for v in cells.values() for c in v})
    assert {s: {str(r): labels[c] for r, c in enumerate(v)} for s, v in cells.items()} == want
