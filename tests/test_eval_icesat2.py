import csv
import json

import numpy as np
import pytest

from canopy2ground.evaluation import icesat2
from canopy2ground.evaluation.icesat2 import overpass_debiased, score


def _segments(n_per=60, seed=0):
    """Three overpasses over a sloping surface, each with its own datum offset."""
    rng = np.random.default_rng(seed)
    rgt = np.repeat(np.array(["101", "202", "303"]), n_per)
    cycle = np.repeat(np.array(["4", "4", "5"]), n_per)
    truth = rng.uniform(50, 900, rgt.size)
    offset = np.repeat([14.1, 15.0, 13.7], n_per)
    seg = {"h_mean": truth + offset + rng.normal(0, 0.3, rgt.size),
           "rms_misfit": np.full(rgt.size, 0.5), "n_fit_photons": np.full(rgt.size, 40.0),
           "h_prior": truth + rng.normal(0, 3.0, rgt.size), "slope_deg": rng.uniform(0, 40, rgt.size),
           "rgt": rgt, "cycle": cycle}
    return seg, truth


def test_debiasing_removes_each_overpass_offset():
    overpass = np.repeat(["a", "b"], 5)
    truth = np.arange(10.0)
    h_icesat = truth + np.repeat([10.0, -3.0], 5)
    e = overpass_debiased(h_icesat, truth, overpass)
    assert np.allclose(e, 0.0)
    e = overpass_debiased(h_icesat, truth + 1.0, overpass)      # a constant surface bias is also absorbed
    assert np.allclose(e, 0.0)


def test_score_gates_and_pairs():
    seg, truth = _segments()
    seg["rms_misfit"][0] = 2.5            # fails the fit-quality gate
    seg["n_fit_photons"][1] = 12          # too few photons
    seg["h_prior"][2] = 0.0               # sea fill: land gate
    good = truth + np.random.default_rng(1).normal(0, 0.5, truth.size)
    good[3] = np.nan                      # off this map: dropped for every surface
    res = score(seg, {"good": good, "noisy": truth + np.random.default_rng(2).normal(0, 2.0, truth.size)})

    assert (res["n_segments"], res["n_gated"], res["n_land"], res["n_common"]) == (180, 178, 177, 176)
    assert res["n_overpasses"] == 3
    assert list(res["surfaces"]) == ["prior", "good", "noisy"]
    assert list(res["paired"]) == ["good_vs_prior", "noisy_vs_prior", "noisy_vs_good"]
    s = res["surfaces"]
    assert s["good"]["median_abs_err_m"] < s["noisy"]["median_abs_err_m"] < s["prior"]["median_abs_err_m"]
    assert res["paired"]["good_vs_prior"]["all"]["lo"] > 0
    assert res["paired"]["noisy_vs_good"]["all"]["point"] < 0
    steep = res["paired"]["good_vs_prior"]["steep_top_quartile"]
    assert steep["n_segments"] == pytest.approx(176 / 4, abs=1)


def test_score_keep_filter():
    seg, truth = _segments()
    keep = np.arange(truth.size) % 2 == 0
    res = score(seg, {"m": truth}, keep)
    assert res["n_common"] == keep.sum()


def _write(path, z, nodata=None):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    tr = from_origin(750000.0, 2300000.0, 1.0, 1.0)
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype=z.dtype,
                       crs="EPSG:6634", transform=tr, nodata=nodata) as dst:
        dst.write(z, 1)


def _plane_raster(path, nodata=None):
    """z = 100 + 2 col - 0.5 row on a 1 m grid; one nodata pixel at (10, 10) if ``nodata`` is given."""
    rows, cols = np.mgrid[0:40, 0:50]
    z = (100.0 + 2.0 * cols - 0.5 * rows).astype(np.float32)
    if nodata is not None:
        z[10, 10] = nodata
    _write(path, z, nodata)


def _lonlat(x, y):
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:6634", "EPSG:4326", always_xy=True)
    return tf.transform(np.asarray(x), np.asarray(y))


def test_bilinear_is_exact_on_a_plane(tmp_path):
    _plane_raster(tmp_path / "plane.tif", nodata=-9999.0)
    x = np.array([750012.3, 750030.75, 749990.0, 750010.5])      # inside, inside, off-raster, touches nodata
    y = np.array([2299980.6, 2299970.25, 2299980.0, 2299989.5])
    lon, lat = _lonlat(x, y)
    h = icesat2.bilinear(tmp_path / "plane.tif", lon, lat)
    col, row = x - 750000.0 - 0.5, 2300000.0 - y - 0.5          # pixel-centre convention
    expect = 100.0 + 2.0 * col - 0.5 * row
    assert h[:2] == pytest.approx(expect[:2], abs=1e-6)
    assert np.isnan(h[2]) and np.isnan(h[3])
    same = icesat2.bilinear(tmp_path / "plane.tif", x, y, crs="EPSG:6634")
    assert same[:2] == pytest.approx(h[:2], abs=1e-6)


def test_fill_floor_raises_prior_filled_pixels(tmp_path):
    _write(tmp_path / "map.tif", np.full((20, 20), 0.2, np.float32))
    flags = np.zeros((20, 20), np.uint8)
    flags[:, 10:] = 2                                           # right half filled from the prior
    _write(tmp_path / "flags.tif", flags)
    x, y = np.array([750004.2, 750015.2]), np.array([2299990.3, 2299990.3])
    h = icesat2.bilinear(tmp_path / "map.tif", x, y, crs="EPSG:6634", floor_flags=tmp_path / "flags.tif")
    assert h == pytest.approx([0.2, 1.0])


def test_outside_mask(tmp_path):
    pytest.importorskip("shapely")
    sq = [[-156.2, 20.7], [-156.1, 20.7], [-156.1, 20.8], [-156.2, 20.8], [-156.2, 20.7]]
    feature = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [sq]}}
    gj = {"type": "FeatureCollection", "features": [feature]}
    (tmp_path / "fp.geojson").write_text(json.dumps(gj))
    lon, lat = np.array([-156.15, -156.05]), np.array([20.75, 20.75])
    out = icesat2.outside_mask(lon, lat, tmp_path / "fp.geojson")
    assert out.tolist() == [False, True]


def test_score_cli_end_to_end(tmp_path):
    _plane_raster(tmp_path / "map.tif")
    rng = np.random.default_rng(0)
    x = rng.uniform(750002, 750047, 90)
    y = rng.uniform(2299962, 2299997, 90)
    lon, lat = _lonlat(x, y)
    col, row = x - 750000.5, 2300000.0 - y - 0.5
    truth = 100.0 + 2.0 * col - 0.5 * row
    with open(tmp_path / "seg.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "h_mean", "rms_misfit", "n_fit_photons", "rgt", "cycle", "h_prior",
                    "slope_deg", "h_old"])
        for i in range(90):                                     # overpass 100 + i % 3, offset 14 + i % 3
            w.writerow([lon[i], lat[i], truth[i] + 14.0 + (i % 3), 0.4, 30, 100 + i % 3, 7,
                        truth[i] + rng.normal(0, 2), rng.uniform(0, 30), truth[i] + rng.normal(0, 1)])
    out = tmp_path / "res.json"
    icesat2.main(["score", "--segments", str(tmp_path / "seg.csv"), "--column", "old=h_old",
                  "--surface", f"new={tmp_path / 'map.tif'}", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["n_common"] == 90 and res["n_overpasses"] == 3
    assert res["surfaces"]["new"]["median_abs_err_m"] < 1e-6
    assert list(res["paired"]) == ["old_vs_prior", "new_vs_prior", "new_vs_old"]
