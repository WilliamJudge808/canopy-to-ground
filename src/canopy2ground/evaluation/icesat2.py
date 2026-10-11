"""ICESat-2 check of DTMs over terrain with no lidar.

ICESat-2 crosses East Maui, where no lidar exists, so it is the only
independent ground truth there. The protocol was fixed before the first
comparison ran:

* Segments are ATL06-SR fits to ATL08 ground photons (SlideRule ``atl06p``:
  40 m segments every 20 m, at least 10 photons over at least 10 m, land
  surface type). Quality gates: fit rms <= 2 m and >= 20 photons.
* Land gate: the Copernicus prior is exactly 0 (its sea fill) under most
  gated segments, which are sea-surface returns; those are dropped.
* Every surface is sampled bilinearly at the segment centre, and all
  surfaces are scored on the segments where every one of them is valid.
* ICESat-2 heights are ellipsoidal and the DTMs geoid-referenced, so within
  each overpass (reference ground track + cycle) the median height difference
  is subtracted, separately for each surface.
* Paired deltas ``|err_base| - |err_better|`` per segment, median, with a
  bootstrap that resamples whole overpasses.

    python -m canopy2ground.evaluation.icesat2 fetch --prior data/priors/copernicus \\
        --out data/icesat/segments.csv
    python -m canopy2ground.evaluation.icesat2 score --segments data/icesat/segments.csv \\
        --surface v3=maps/east_maui_dtm.tif --out results/icesat.json \\
        [--column v1=h_model] [--fill-floor v2=maps/v2_source.tif] [--outside lidar_coverage.geojson]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..grid import CRS
from ..mapping import EAST_MAUI_BBOX
from ..metrics import nmad
from ..prior import PriorRaster
from .stats import cluster_bootstrap

SEGMENT_PARAMS = {"len": 40.0, "res": 20.0, "cnt": 10, "ats": 10.0, "atl08_class": ["atl08_ground"]}
MAX_RMS_M = 2.0
MIN_PHOTONS = 20
SLOPE_STEP_M = 30.0                                  # half-baseline of the prior slope, one GLO-30 cell
FLOOR_FLAG = 2                                       # source-flag value of prior-filled map pixels


def bilinear(raster, x: np.ndarray, y: np.ndarray, crs: str = "EPSG:4326",
             floor_flags: str | Path | None = None, floor: float = 1.0) -> np.ndarray:
    """Bilinear sample of band 1 at points given in ``crs`` (pixel-centre convention).

    ``raster`` is a path or an open rasterio dataset. Off-raster points and
    nodata give NaN. ``floor_flags`` is a map's source-flag raster: where any
    of the 3 x 3 flag cells around a point marks a prior-filled pixel, the
    2 x 2 window is floored at ``floor`` m before interpolation, as the
    published map is.
    """
    import rasterio
    from pyproj import Transformer

    ds = rasterio.open(raster) if isinstance(raster, (str, Path)) else raster
    flags = rasterio.open(floor_flags) if floor_flags else None
    try:
        xs, ys = Transformer.from_crs(crs, ds.crs, always_xy=True).transform(x, y)
        inv = ~ds.transform
        out = np.full(len(xs), np.nan)
        for k, (xx, yy) in enumerate(zip(xs, ys)):
            c, r = inv * (xx, yy)
            c, r = c - 0.5, r - 0.5
            c0, r0 = int(np.floor(c)), int(np.floor(r))
            if c0 < 0 or r0 < 0 or c0 + 1 >= ds.width or r0 + 1 >= ds.height:
                continue
            w = ds.read(1, window=((r0, r0 + 2), (c0, c0 + 2))).astype(np.float64)
            if ds.nodata is not None and not np.isnan(ds.nodata):
                w[w == ds.nodata] = np.nan
            if flags is not None:
                fc, fr = ~flags.transform * (xx, yy)
                fl = flags.read(1, window=((int(fr) - 1, int(fr) + 2), (int(fc) - 1, int(fc) + 2)))
                if (fl == FLOOR_FLAG).any():
                    w = np.where(np.isfinite(w) & (w < floor), floor, w)
            dc, dr = c - c0, r - r0
            out[k] = (w[0, 0] * (1 - dc) * (1 - dr) + w[0, 1] * dc * (1 - dr)
                      + w[1, 0] * (1 - dc) * dr + w[1, 1] * dc * dr)
    finally:
        if flags is not None:
            flags.close()
        if ds is not raster:
            ds.close()
    return out


def overpass_debiased(h_icesat: np.ndarray, h_surface: np.ndarray, overpass: np.ndarray) -> np.ndarray:
    """Height error ``h_icesat - h_surface`` minus its median within each overpass."""
    d = h_icesat - h_surface
    labels, inv = np.unique(overpass, return_inverse=True)
    med = np.array([np.median(d[inv == i]) for i in range(len(labels))])
    return d - med[inv]


def error_summary(e: np.ndarray) -> dict:
    return {"median_abs_err_m": float(np.median(np.abs(e))), "nmad_m": nmad(e),
            "le90_m": float(np.quantile(np.abs(e), 0.9))}


def outside_mask(lon: np.ndarray, lat: np.ndarray, geojson: str | Path) -> np.ndarray:
    """True for points outside every polygon of a lon/lat GeoJSON (e.g. a lidar footprint)."""
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    from shapely.prepared import prep

    gj = json.loads(Path(geojson).read_text(encoding="utf-8"))
    poly = prep(unary_union([shape(f["geometry"]).buffer(0) for f in gj["features"]]))
    return np.array([not poly.contains(Point(x, y)) for x, y in zip(lon, lat)])


def land_segments(seg: dict[str, np.ndarray],
                  keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment masks: (passes the quality gates, and is also on land and inside ``keep``)."""
    gated = (seg["rms_misfit"] <= MAX_RMS_M) & (seg["n_fit_photons"] >= MIN_PHOTONS)
    land = gated & (seg["h_prior"] != 0)
    if keep is not None:
        land &= keep
    return gated, land


def score(seg: dict[str, np.ndarray], surfaces: dict[str, np.ndarray],
          keep: np.ndarray | None = None) -> dict:
    """Score sampled surfaces against ICESat-2 segments.

    ``seg`` holds per-segment arrays (h_mean, rms_misfit, n_fit_photons, h_prior,
    slope_deg, rgt, cycle). ``surfaces`` maps a name to heights at the same
    segments; the prior is always scored first, as ``prior``. ``keep`` is an
    extra segment filter applied with the land gate.
    """
    gated, land = land_segments(seg, keep)
    h = {"prior": seg["h_prior"], **surfaces}
    ok = land & np.logical_and.reduce([np.isfinite(v) for v in h.values()])
    overpass = np.char.add(np.char.add(seg["rgt"].astype(str), "_"), seg["cycle"].astype(str))[ok]
    err = {name: overpass_debiased(seg["h_mean"][ok], v[ok], overpass) for name, v in h.items()}

    res = {"n_segments": int(len(gated)), "n_gated": int(gated.sum()), "n_land": int(land.sum()),
           "n_common": int(ok.sum()), "n_overpasses": int(len(np.unique(overpass))),
           "surfaces": {name: error_summary(e) for name, e in err.items()}, "paired": {}}
    names = list(err)
    pairs = [(n, "prior") for n in names[1:]] + list(zip(names[2:], names[1:]))
    slope = seg["slope_deg"][ok]
    steep = slope >= np.quantile(slope, 0.75)
    for better, base in pairs:
        d = np.abs(err[base]) - np.abs(err[better])          # positive: `better` is closer to ICESat-2
        res["paired"][f"{better}_vs_{base}"] = {
            "all": cluster_bootstrap(d, overpass),
            "steep_top_quartile": cluster_bootstrap(d[steep], overpass[steep]),
        }
    return res


def read_segments(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} holds no segments")
    out = {}
    for k in rows[0]:
        col = np.array([r[k] for r in rows])
        try:
            out[k] = col.astype(float) if k not in ("rgt", "cycle", "time") else col
        except ValueError:
            out[k] = col
    return out


def _sliderule_with_user_agent() -> None:
    """Make SlideRule's module-level session send ``USER_AGENT``.

    The client has no public way to set request headers, and ``icesat2.atl06p``
    always uses the private global ``sliderule.sliderule.slideruleSession``, so
    that global is replaced. Written against sliderule 5.x; recheck on upgrade.
    """
    import importlib

    from sliderule.session import Session

    from ..sources.http import USER_AGENT

    session = Session()
    session.session.headers["User-Agent"] = USER_AGENT
    importlib.import_module("sliderule.sliderule").slideruleSession = session


def fetch(bbox: tuple[float, float, float, float], prior: str | Path, out: Path) -> int:
    """Query SlideRule for ground segments in ``bbox``; write them with the prior height and slope."""
    from pyproj import Transformer
    from sliderule import icesat2

    _sliderule_with_user_agent()
    x0, y0, x1, y1 = bbox
    poly = [{"lon": x0, "lat": y0}, {"lon": x1, "lat": y0}, {"lon": x1, "lat": y1},
            {"lon": x0, "lat": y1}, {"lon": x0, "lat": y0}]
    gdf = icesat2.atl06p({"poly": poly, "srt": icesat2.SRT_LAND, **SEGMENT_PARAMS})
    lon, lat = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()

    pr = PriorRaster(prior, CRS)
    try:
        h_prior = bilinear(pr.src, lon, lat)
        # slope stratifier from the prior, so it is independent of every scored model
        x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
        s = SLOPE_STEP_M
        east, west = bilinear(pr.src, x + s, y, crs=CRS), bilinear(pr.src, x - s, y, crs=CRS)
        north, south = bilinear(pr.src, x, y + s, crs=CRS), bilinear(pr.src, x, y - s, crs=CRS)
    finally:
        pr.close()
    slope = np.degrees(np.arctan(np.hypot((east - west) / (2 * s), (north - south) / (2 * s))))

    cols = ["h_mean", "rms_misfit", "n_fit_photons", "rgt", "cycle", "spot", "gt"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "lon", "lat", *cols, "h_prior", "slope_deg"])
        for i, t in enumerate(gdf.index):
            w.writerow([t, lon[i], lat[i], *(gdf[c].iloc[i] for c in cols), h_prior[i], slope[i]])
    return len(gdf)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download ground segments and sample the prior at them")
    f.add_argument("--bbox", type=float, nargs=4, default=EAST_MAUI_BBOX, metavar=("W", "S", "E", "N"))
    f.add_argument("--prior", required=True, help="Copernicus GeoTIFF or directory of tiles")
    f.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("score", help="score DTMs against the segments")
    s.add_argument("--segments", required=True, type=Path, help="CSV written by `fetch`")
    s.add_argument("--surface", nargs="*", default=[], metavar="NAME=RASTER")
    s.add_argument("--column", nargs="*", default=[], metavar="NAME=COLUMN",
                   help="surfaces already sampled into the CSV (scored before --surface)")
    s.add_argument("--fill-floor", nargs="*", default=[], metavar="NAME=FLAGS",
                   help="source-flag raster of a map whose prior-filled pixels are floored at 1 m")
    s.add_argument("--outside", type=Path, help="GeoJSON (lon/lat): score only segments outside it")
    s.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        print(f"wrote {a.out}: {fetch(tuple(a.bbox), a.prior, a.out)} segments")
        return

    seg = read_segments(a.segments)
    keep = outside_mask(seg["lon"], seg["lat"], a.outside) if a.outside else None
    _, todo = land_segments(seg, keep)                       # sample only what can be scored
    surfaces = {name: seg[col] for name, col in (c.split("=", 1) for c in a.column)}
    floors = dict(c.split("=", 1) for c in a.fill_floor)
    for spec in a.surface:
        name, path = spec.split("=", 1)
        h = np.full(len(todo), np.nan)
        h[todo] = bilinear(path, seg["lon"][todo], seg["lat"][todo], floor_flags=floors.get(name))
        surfaces[name] = h
        print(f"sampled {name}: valid {np.isfinite(h[todo]).mean():.3f} of {int(todo.sum())} land segments")
    res = score(seg, surfaces, keep)
    res["outside"] = a.outside.name if a.outside else None

    print(f"segments: {res['n_gated']} gated -> {res['n_land']} land -> {res['n_common']} valid on every "
          f"surface, {res['n_overpasses']} overpasses")
    for name, r in res["surfaces"].items():
        print(f"  {name:10s} median |err| {r['median_abs_err_m']:.3f} m  NMAD {r['nmad_m']:.3f}  "
              f"LE90 {r['le90_m']:.3f}")
    for pair, r in res["paired"].items():
        al, st = r["all"], r["steep_top_quartile"]
        print(f"  {pair}: {al['point']:+.3f} m [{al['lo']:+.3f}, {al['hi']:+.3f}]   "
              f"steep quartile {st['point']:+.3f} [{st['lo']:+.3f}, {st['hi']:+.3f}]")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
