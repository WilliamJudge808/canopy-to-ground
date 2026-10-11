"""Same-chip comparison against public bare-earth DEMs on lidar.

Products, both corrected versions of Copernicus GLO-30 at 1 arcsec, EGM2008:

* ``fabdem``: FABDEM v1-2 (Hawker et al. 2022), forests and buildings removed
  by random forest. CC BY-NC-SA 4.0: research use only. ``download`` fetches
  it from the University of Bristol into the local priors directory; never
  commit or redistribute the tiles or the arrays built from them.
* ``gedtm30``: GEDTM30 (Ho et al. 2025, OpenLandMap), an ensemble DTM fitted
  to ICESat-2 and GEDI ground heights. CC BY 4.0. Stored as int32 decimetres.

``build`` warps each product exactly like the Copernicus prior (lanczos onto
the 512 x 512 chip grid, no datum fit) into ``dem_<product>.npy`` /
``dmask_<product>.npy`` next to the cache arrays; ``verify-grid`` re-warps
Copernicus with the same code and compares it with the stored prior.

``score`` puts every surface on the SAME pixels of each chip (lidar valid and
every product valid) and reports, per chip:

* 1 m: mean |surface - lidar|;
* 30 m: surface and lidar both averaged over 30 x 30 px blocks with >= 50%
  valid pixels, then mean |difference|;
* slope strata 0-5 / 5-10 / 10-20 / >20 deg (slope of the prior at 1 m): 1 m
  MAE inside the stratum, for chips with >= 256 px in it.

Medians and means over chips, and paired per-chip differences (product minus
model, positive = model better) with a bootstrap over chips and, since chips
of one split block are not independent, over whole blocks
(``evaluation.blocks``, from the cache's ``meta.json``). Without
``--checkpoint`` only the products and Copernicus are scored.

    python -m canopy2ground.evaluation.external download --priors data/priors
    python -m canopy2ground.evaluation.external build --cache data/cache --priors data/priors --split val
    python -m canopy2ground.evaluation.external build --cache data/cache --priors data/priors --split train \\
        --regions molokai_wet
    python -m canopy2ground.evaluation.external score --cache data/cache --layer-root data/layers \\
        --split val --checkpoint m1.pt ... m6.pt --fused-attention --out results/external_val.json
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from ..build.cache import chip_ids
from ..data.dataset import CHIP, BatchOptions, ChipDataset, DataConfig, make_batch
from ..grid import CRS, chip_bounds
from .blocks import products_vs_model
from .stats import bootstrap_mean

PRODUCTS = ("fabdem", "gedtm30")
SLOPE_BINS = ((0, 5), (5, 10), (10, 20), (20, 90))
BLOCK = 30
MIN_PIXELS = 256
METRICS = ("mae1", "mae30", *(f"bin{lo}_{hi}" for lo, hi in SLOPE_BINS))

# Maui County, padded so the lanczos kernel never reads past the window edge
COUNTY_BBOX = (-157.4, 20.4, -155.9, 21.3)
PAD_DEG = 0.05
FABDEM_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/N20W160-N30W150_FABDEM_V1-2.zip"
FABDEM_TILES = ("N20W158", "N20W157", "N20W156", "N21W158", "N21W157")
GEDTM_URL = ("https://s3.opengeohub.org/global/edtm/"
             "gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v20250611.tif")
GEDTM_FILE = "gedtm30/gedtm30_maui_county_v20250611.tif"
GEDTM_SCALE = 0.1                                   # int32 decimetres -> metres


# ------------------------------------------------------------------ building the product arrays

def download(priors: Path) -> None:
    """FABDEM tiles over the County, and a County window of GEDTM30 (read remotely, not the global file)."""
    import rasterio
    from rasterio.windows import from_bounds

    from ..sources.http import USER_AGENT
    from ..sources.raster import gdal_env

    fd = priors / "fabdem"
    fd.mkdir(parents=True, exist_ok=True)
    z = fd / Path(FABDEM_URL).name
    if not z.exists():
        req = urllib.request.Request(FABDEM_URL, headers={"User-Agent": USER_AGENT})
        part = z.with_suffix(".part")
        with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(part, z)
    with zipfile.ZipFile(z) as zf:
        for t in FABDEM_TILES:
            zf.extract(f"{t}_FABDEM_V1-2.tif", fd)

    out = priors / GEDTM_FILE
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with gdal_env(), rasterio.open("/vsicurl/" + GEDTM_URL) as src:
        x0, y0, x1, y1 = COUNTY_BBOX
        win = from_bounds(x0 - PAD_DEG, y0 - PAD_DEG, x1 + PAD_DEG, y1 + PAD_DEG,
                          src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        profile = src.profile.copy()
        profile.update(driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       transform=src.window_transform(win), compress="deflate", predictor=2, tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="NO")
    part = out.with_suffix(".part.tif")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(source_url=GEDTM_URL, value_scale=str(GEDTM_SCALE), units="decimetres_EGM2008")
    os.replace(part, out)


def _float_dataset(arr: np.ndarray, transform, crs):
    from rasterio.io import MemoryFile

    mem = MemoryFile()
    ds = mem.open(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                  crs=crs, transform=transform, nodata=np.nan)
    ds.write(arr.astype(np.float32), 1)
    return ds, mem


def open_product(product: str, priors: Path):
    """The product as one in-memory float32 metres dataset with NaN nodata. Returns (dataset, memfile)."""
    import rasterio
    from rasterio.merge import merge

    if product == "fabdem":
        tiles = [rasterio.open(priors / "fabdem" / f"{t}_FABDEM_V1-2.tif") for t in FABDEM_TILES]
        nodata, crs = tiles[0].nodata, tiles[0].crs
        arr, transform = merge(tiles, nodata=nodata)
        for t in tiles:
            t.close()
        a = arr[0].astype(np.float32)
        if nodata is not None:
            a[a == nodata] = np.nan
        return _float_dataset(a, transform, crs)
    if product == "gedtm30":
        with rasterio.open(priors / GEDTM_FILE) as s:
            raw = s.read(1)
            a = raw.astype(np.float32) * GEDTM_SCALE
            if s.nodata is not None:
                a[raw == s.nodata] = np.nan
            return _float_dataset(a, s.transform, s.crs)
    raise ValueError(f"unknown product {product!r}; expected one of {PRODUCTS}")


def warp_chip(src, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """(elevation, valid) of ``src`` warped onto one chip, lanczos as for the Copernicus prior."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    dst = np.full((CHIP, CHIP), np.nan, np.float32)
    reproject(rasterio.band(src, 1), dst, dst_transform=from_bounds(*chip_bounds(stem), CHIP, CHIP),
              dst_crs=CRS, resampling=Resampling.lanczos, src_nodata=src.nodata, dst_nodata=np.nan)
    good = np.isfinite(dst)
    if src.nodata is not None and not np.isnan(src.nodata):
        good &= dst != src.nodata
    return dst, good


def _rows(cache: Path, split: str, n: int, regions) -> list[int]:
    if not regions:
        return list(range(n))
    labels = json.loads((cache / "island_labels.json").read_text(encoding="utf-8-sig"))[split]
    return sorted(int(k) for k, v in labels.items() if v in set(regions))


def build(product: str, cache: Path, priors: Path, split: str, regions=None) -> None:
    """Write ``dem_<product>.npy`` (float32 m, voids 0) and ``dmask_<product>.npy`` for a cache split.

    Arrays keep the full (n, 512, 512) shape so rows line up with the cache;
    rows outside ``regions`` are never written and stay sparse on disk.
    """
    d = cache / split
    stems = chip_ids(d)
    n = len(stems)
    rows = _rows(cache, split, n, regions)
    tmp_dem, tmp_mask = d / f".dem_{product}.building.npy", d / f".dmask_{product}.building.npy"
    out = np.lib.format.open_memmap(tmp_dem, mode="w+", dtype=np.float32, shape=(n, CHIP, CHIP))
    msk = np.lib.format.open_memmap(tmp_mask, mode="w+", dtype=bool, shape=(n, CHIP, CHIP))
    src, mem = open_product(product, priors)
    n_void = 0
    try:
        for j, k in enumerate(rows):
            dem, good = warp_chip(src, stems[k])
            n_void += int(not good.all())
            out[k] = np.where(good, dem, 0.0)
            msk[k] = good
            if (j + 1) % 200 == 0:
                print(f"  {split}/{product}: {j + 1}/{len(rows)}", flush=True)
    finally:
        src.close()
        mem.close()
    out.flush()
    msk.flush()
    del out, msk
    os.replace(tmp_dem, d / f"dem_{product}.npy")
    os.replace(tmp_mask, d / f"dmask_{product}.npy")
    print(f"  {split}/{product}: {len(rows)}/{n} rows, {n_void} with voids", flush=True)


def verify_grid(cache: Path, priors: Path, k: int = 6, seed: int = 0) -> float:
    """Re-warp Copernicus with ``warp_chip`` on ``k`` random val chips; returns max |new - stored| in m."""
    from ..prior import PriorRaster

    d = cache / "val"
    stems = chip_ids(d)
    ref = np.load(d / "dem_copernicus.npy", mmap_mode="r")
    ref_mask = np.load(d / "dmask_copernicus.npy", mmap_mode="r")
    prior = PriorRaster(priors / "copernicus", CRS)
    worst = 0.0
    try:
        for r in sorted(np.random.default_rng(seed).choice(len(stems), k, replace=False)):
            dem, good = warp_chip(prior.src, stems[r])
            m = good & np.asarray(ref_mask[r])
            diff = float(np.abs(np.where(good, dem, 0.0)[m] - np.asarray(ref[r], np.float32)[m]).max())
            agree = float(np.mean(good == np.asarray(ref_mask[r])))
            print(f"  row {r} {stems[r]}: max |new - stored| {diff:.4f} m, mask agreement {agree:.4f}")
            worst = max(worst, diff)
    finally:
        prior.close()
    return worst


# ------------------------------------------------------------------ scoring

def block_mae(p: np.ndarray, t: np.ndarray, v: np.ndarray, b: int = BLOCK) -> float:
    """MAE between ``b`` x ``b`` block means of ``p`` and ``t`` over blocks at least half valid."""
    h, w = (p.shape[0] // b) * b, (p.shape[1] // b) * b

    def blk(a):
        return a[:h, :w].reshape(h // b, b, w // b, b)

    n = blk(v).astype(np.float64).sum((1, 3))
    ok = n >= 0.5 * b * b
    if not ok.any():
        return np.nan
    ps = (blk(np.where(v, p, 0)).sum((1, 3)) / np.maximum(n, 1))[ok]
    ts = (blk(np.where(v, t, 0)).sum((1, 3)) / np.maximum(n, 1))[ok]
    return float(np.abs(ps - ts).mean())


def chip_metrics(surfaces: dict[str, np.ndarray], lidar: np.ndarray, valid: np.ndarray,
                 prior: np.ndarray) -> dict[str, dict[str, float]]:
    """Every metric of every surface on one chip, on the common ``valid`` pixels."""
    gy, gx = np.gradient(prior)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    out = {}
    for s, z in surfaces.items():
        e = np.abs(z - lidar)
        row = {"mae1": float(e[valid].mean()), "mae30": block_mae(z, lidar, valid)}
        for lo, hi in SLOPE_BINS:
            sel = valid & (slope >= lo) & (slope < hi)
            row[f"bin{lo}_{hi}"] = float(e[sel].mean()) if sel.sum() >= MIN_PIXELS else np.nan
        out[s] = row
    return out


def score(ds: ChipDataset, products: dict, model=None, batch_size: int = 4, device="cuda",
          log_every: int = 200) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    """Per-chip metrics of the model (if given), Copernicus and each product. Returns (rows, per)."""
    from ..metrics import predict_batch

    surfaces = (("model",) if model is not None else ()) + ("copernicus",) + tuple(products)
    per = {s: {m: [] for m in METRICS} for s in surfaces}
    rows = []
    for i in range(0, len(ds), batch_size):
        b = make_batch(ds, list(range(i, min(i + batch_size, len(ds)))), BatchOptions(), device=device)
        if b is None:
            continue
        prior = b["dem"].cpu().numpy()
        if model is not None:
            res = predict_batch(model, b)
            pred = (b["dem"] + res * b["in_mask"]).cpu().numpy()
        lidar = b["lidar"].cpu().numpy()
        mask = (b["mask"] > 0.5).cpu().numpy()
        for k, row in enumerate(b["chip_ids"]):
            surf = {"model": pred[k]} if model is not None else {}
            surf["copernicus"] = prior[k]
            v = mask[k].copy()
            for name, (dem, dmask) in products.items():
                surf[name] = np.asarray(dem[row], dtype=np.float32)
                v &= np.asarray(dmask[row]).astype(bool)
            if v.sum() < MIN_PIXELS:
                continue
            rows.append(int(row))
            for s, ms in chip_metrics(surf, lidar[k], v, prior[k]).items():
                for m, x in ms.items():
                    per[s][m].append(x)
        if log_every and i % log_every == 0:
            print(f"  {i}/{len(ds)} chips, {len(rows)} scored", flush=True)
    return rows, per


def summarise(rows: list[int], per: dict, ids: list[str] | None = None) -> dict:
    """Per-surface medians and means, and the model's paired chip bootstraps.

    Given the cache's chip ``ids`` (row order), the paired 1 m differences are
    also bootstrapped over whole split blocks, as ``block_bootstrap_mae1``.
    """
    surfaces = list(per)
    res = {"n_chips": len(rows), "surfaces": {}, "paired_product_minus_model": {},
           "per_chip_mae1": {s: dict(zip(rows, per[s]["mae1"])) for s in surfaces}}
    for s in surfaces:
        res["surfaces"][s] = {m: {"median": float(np.nanmedian(v)), "mean": float(np.nanmean(v)),
                                  "n": int(np.isfinite(v).sum())}
                              for m, v in ((m, np.array(x, float)) for m, x in per[s].items())}
    if "model" in per:
        for s in surfaces[1:]:
            res["paired_product_minus_model"][s] = {
                m: bootstrap_mean(np.array(per[s][m], float) - np.array(per["model"][m], float))
                for m in per[s]}
        if ids is not None:
            res["block_bootstrap_mae1"] = products_vs_model(res["per_chip_mae1"], ids)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch FABDEM and GEDTM30 over Maui County (login node)")
    d.add_argument("--priors", required=True, type=Path)
    v = sub.add_parser("verify-grid", help="re-warp Copernicus with this code and compare with the cache")
    v.add_argument("--cache", required=True, type=Path)
    v.add_argument("--priors", required=True, type=Path)
    b = sub.add_parser("build", help="warp the products onto a cache split")
    b.add_argument("--cache", required=True, type=Path)
    b.add_argument("--priors", required=True, type=Path)
    b.add_argument("--split", default="val")
    b.add_argument("--regions", nargs="*", help="only these region labels (island_labels.json)")
    b.add_argument("--product", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    s = sub.add_parser("score", help="same-chip scores of the model, Copernicus and the products")
    s.add_argument("--cache", required=True, type=Path)
    s.add_argument("--layer-root", type=Path)
    s.add_argument("--split", default="val")
    s.add_argument("--regions", nargs="*")
    s.add_argument("--checkpoint", nargs="*", default=[], help="ensemble members (none: products only)")
    s.add_argument("--fused-attention", action="store_true")
    s.add_argument("--device", default="cuda")
    s.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "download":
        download(a.priors)
    elif a.cmd == "verify-grid":
        print(f"max |new - stored| {verify_grid(a.cache, a.priors):.4f} m")
    elif a.cmd == "build":
        for p in a.product:
            build(p, a.cache, a.priors, a.split, a.regions)
    else:
        _score_main(a)


def _score_main(a) -> None:
    from ..models.speed import use_fused_mit_attention
    from .ensemble import Ensemble, load_members

    model, layers = None, ()
    if a.checkpoint:
        if a.fused_attention:
            use_fused_mit_attention()
        members, layers = load_members(a.checkpoint, a.device)
        model = Ensemble(members).eval()
    cfg = DataConfig(root=str(a.cache), layer_root=str(a.layer_root) if a.layer_root else None, layers=layers,
                     coast=model is not None, regions=tuple(a.regions) if a.regions else None)
    ds = ChipDataset(cfg, a.split)
    split_dir = a.cache / a.split
    products = {p: (np.load(split_dir / f"dem_{p}.npy", mmap_mode="r"),
                    np.load(split_dir / f"dmask_{p}.npy", mmap_mode="r")) for p in PRODUCTS}
    rows, per = score(ds, products, model, device=a.device)
    ids = chip_ids(split_dir) if (split_dir / "meta.json").exists() else None
    res = {"split": a.split, "regions": a.regions, "models": [Path(p).name for p in a.checkpoint],
           **summarise(rows, per, ids)}

    print(f"\n{res['n_chips']} chips ({a.split}{' ' + ','.join(a.regions) if a.regions else ''}), "
          "median/mean of per-chip MAE in m")
    for m in METRICS:
        cells = (f"{s} {r[m]['median']:.3f}/{r[m]['mean']:.3f}" for s, r in res["surfaces"].items())
        print(f"  {m:9s} " + "  ".join(cells))
    for s, q in res["paired_product_minus_model"].items():
        for m in ("mae1", "mae30"):
            r = q[m]
            print(f"  {s:10s} - model, {m}: mean {r['mean']:+.3f} m [{r['lo']:+.3f}, {r['hi']:+.3f}], "
                  f"model better on {r['frac_positive']:.0%} of {r['n']} chips")
    for s, r in res.get("block_bootstrap_mae1", {}).items():
        print(f"  {s:10s} - model, mae1 by block: mean {r['mean']:+.3f} m [{r['lo']:+.3f}, {r['hi']:+.3f}], "
              f"model better in {r['blocks_better']}/{r['n_blocks']} blocks")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
