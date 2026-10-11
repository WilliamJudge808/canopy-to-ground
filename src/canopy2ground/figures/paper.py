"""The publication figures: single-column journal layout (4.32 x 5.4 in) at 250 dpi = 1080 x 1350 px.

    python -m canopy2ground.figures.paper --panel figs/hanawi_gorge.npz --overview figs/overview.npz \\
        --results results --out figs

Journal point sizes at this size read at phone size (~30 px body text).

1. ``before_after``  Copernicus vs this model over one gorge.
2. ``region_map``    East Maui: reconstructed area, lidar coverage, the gorge's location.
3. ``full_figure``   photo / Copernicus / model strips and an elevation profile.
4. ``accuracy``      median error against three independent references.

Inputs come from ``figures.panels``, and the accuracy figure's numbers from
the evaluation outputs in ``--results``. Every figure that shows Copernicus
carries the attribution notice its licence requires.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from .panels import FIGURE_PLACE

# The first installed of these, not the list itself in rcParams: matplotlib logs a warning for each missing
# family every time it lays out text. Liberation Sans has Arial's metrics; DejaVu Sans ships with matplotlib.
SANS = ("Arial", "Liberation Sans", "DejaVu Sans")
FONT = next(f for f in SANS if f in {e.name for e in font_manager.fontManager.ttflist})
STYLE = {
    "font.family": FONT, "font.size": 8.5, "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.direction": "out", "ytick.direction": "out",
    "axes.titlesize": 8.5, "axes.titlelocation": "left", "legend.frameon": False,
    "savefig.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.bf": f"{FONT}:bold",
}
FIG, DPI = (4.32, 5.4), 250
OURS, OTHER, TRANSECT = "#1f4e79", "#a3a3a3", "#c0392b"
COPERNICUS = ("Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space "
              "GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.")
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
PLACE_NAME = "Kapaula and Hanawī gulches, near Nāhiku"
PLACES = [["Haleakalā", -156.2534, 20.7096, "left"], ["Hāna", -155.988, 20.757, "right"],
          ["Keʻanae", -156.1455, 20.8625, "left"], ["Kīpahulu", -156.0553, 20.6547, "left"]]
EARTH_R = 6378137.0
# Evaluation outputs the accuracy figure reads, and the name the published map has in the ICESat-2 one.
VAL, ICESAT, MOLOKAI = "external_val.json", "icesat_v3_nolidar.json", "external_molokai.json"
ICESAT_MAP = "v3"
styled = mpl.rc_context(STYLE)


def accuracy_groups(results: Path) -> tuple[list, float]:
    """The accuracy figure's panels from the evaluation JSONs in ``results``, and the val win rate.

    Panels are (title, sample, x label, [(method, median error m)]). The win
    rate is the smallest share of validation chips on which the model beats a
    product.
    """
    def load(name: str) -> dict:
        return json.loads((Path(results) / name).read_text(encoding="utf-8"))

    def mae1(r: dict, s: str) -> float:
        return r["surfaces"][s]["mae1"]["median"]

    val, ice, mol = load(VAL), load(ICESAT), load(MOLOKAI)
    groups = [
        ("a  Held-out lidar areas", f"n = {val['n_chips']:,} validation chips",
         "Per-chip MAE, median over chips (m)",
         [("Copernicus GLO-30", mae1(val, "copernicus")), ("FABDEM", mae1(val, "fabdem")),
          ("This model", mae1(val, "model"))]),
        ("b  ICESat-2 laser altimetry, unmapped East Maui",
         f"n = {ice['n_common']:,} ground segments, {ice['n_overpasses']} overpasses",
         "Abs. error, median over segments (m)",
         [("Copernicus GLO-30", ice["surfaces"]["prior"]["median_abs_err_m"]),
          ("This model", ice["surfaces"][ICESAT_MAP]["median_abs_err_m"])]),
        ("c  Unseen rainforest: wet east Molokai",
         f"n = {mol['n_chips']:,} chips, region withheld from training",
         "Per-chip MAE, median over chips (m)",
         [("Copernicus GLO-30", mae1(mol, "copernicus")), ("FABDEM", mae1(mol, "fabdem")),
          ("Same method,\nretrained without it", mae1(mol, "model"))]),
    ]
    wins = min(q["mae1"]["frac_positive"] for q in val["paired_product_minus_model"].values())
    return groups, wins


def caption(fig: Figure, bold: str, rest: str, y: float = 0.012, notice: bool = False) -> None:
    """Bold 'Fig. N | title.' then plain text at 7 pt (~24 px); optional Copernicus notice at 5.6 pt."""
    fig.text(0.04, y + (0.03 if notice else 0), r"$\bf{" + bold.replace(" ", r"\ ") + r"}$ " + rest,
             fontsize=7, ha="left", va="bottom", wrap=True, color="#222222")
    if notice:
        fig.text(0.04, y, COPERNICUS, fontsize=5.6, ha="left", va="bottom", wrap=True, color="#555555")


def panel_label(ax, letter: str, text: str) -> None:
    ax.text(0.015, 0.975, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left",
            path_effects=HALO)
    ax.text(0.075, 0.965, text, transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.5))


def scalebar(ax, metres: float, px_per_m: float = 1.0, xf: float = 0.05, yf: float = 0.1) -> None:
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x = x0 + xf * (x1 - x0)
    y = y0 - yf * (y0 - y1) if y0 > y1 else y0 + (1 - yf) * (y1 - y0)
    w = metres * px_per_m
    ax.plot([x, x + w], [y, y], color="black", lw=2.0, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=3.6, foreground="white")])
    ax.annotate(f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m", (x + w / 2, y), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=7.5, path_effects=HALO)


def hillshade(z: np.ndarray) -> np.ndarray:
    return LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(z), vert_exag=2.0, dx=1.0, dy=1.0)


def shade(h: np.ndarray, lo: float, hi: float, sea: np.ndarray | None = None) -> np.ndarray:
    h = np.clip((h - lo) / (hi - lo), 0, 1)
    rgb = np.dstack([h] * 3) * 0.94 + 0.03
    if sea is not None:
        rgb[sea] = (0.88, 0.92, 0.96)
    return rgb


def deepest_rows(hv: np.ndarray, sea: np.ndarray, height: int) -> int:
    """First row of the full-width band of ``height`` rows with the most relief (the most incised gorges)."""
    gy, gx = np.gradient(hv)
    g = np.abs(gy) + np.abs(gx)
    g[sea] = 0
    return int(np.argmax(np.convolve(g.sum(1), np.ones(height), "valid")))


def _load(npz: Path):
    z = np.load(npz)
    hv, hp = hillshade(z["z_model"]), hillshade(z["z_prior"])
    lo, hi = np.percentile(hv, [1, 99.5])
    return z, hv, hp, lo, hi


def _band(width: int, panel_height: float) -> int:
    """Rows of a full-width strip that fills a 0.92-wide axes of ``panel_height`` (figure fractions)."""
    return int(width * (panel_height * FIG[1]) / (0.92 * FIG[0]))


@styled
def before_after(npz: Path, out: Path, place: str = PLACE_NAME) -> None:
    z, hv, hp, lo, hi = _load(npz)
    sea = z["sea"]
    W = hv.shape[1]
    fig = Figure(figsize=FIG, dpi=DPI)
    ph = 0.405
    band = _band(W, ph)
    r0 = deepest_rows(hv, sea, band)
    sl = (slice(r0, r0 + band), slice(0, W))
    strips = ((shade(hp, lo, hi, sea), "a", "Copernicus GLO-30 (30 m radar DEM)"),
              (shade(hv, lo, hi, sea), "b", "This model (1 m)"))
    for i, (img, letter, text) in enumerate(strips):
        ax = fig.add_axes([0.04, 0.57 - i * (ph + 0.012), 0.92, ph])
        ax.imshow(img[sl], interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        panel_label(ax, letter, text)
        scalebar(ax, 500)
    caption(fig, f"Fig. 1 | {place}, East Maui.",
            "No public lidar covers this ground. Same extent and lighting, 2× vertical exaggeration; "
            "b is a model prediction, not a survey.", notice=True)
    fig.savefig(out, dpi=DPI)


@styled
def full_figure(npz: Path, out: Path, place: str = PLACE_NAME, fig_no: int = 3) -> None:
    z, hv, hp, lo, hi = _load(npz)
    sea, naip = z["sea"], z["naip"]
    W = hv.shape[1]
    fig = Figure(figsize=FIG, dpi=DPI)
    ph = 0.18
    band = _band(W, ph)
    r0 = deepest_rows(hv, sea, band)
    sl = (slice(r0, r0 + band), slice(0, W))
    row = band // 2
    c0, c1 = int(W * 0.03), int(W * 0.97)
    imgs = ((naip, "a", "Aerial photograph (NAIP)"),
            (shade(hp, lo, hi, sea), "b", "Copernicus GLO-30 (30 m)"),
            (shade(hv, lo, hi, sea), "c", "This model (1 m)"))
    for i, (img, letter, text) in enumerate(imgs):
        ax = fig.add_axes([0.04, 0.79 - i * (ph + 0.008), 0.92, ph])
        ax.imshow(img[sl], interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([c0, c1], [row, row], color=TRANSECT, lw=1.0)
        for x, s in ((c0, "A"), (c1, "A′")):
            ax.text(x, row - band * 0.06, s, color=TRANSECT, fontsize=7.5, fontweight="bold", ha="center",
                    va="bottom", path_effects=HALO)
        panel_label(ax, letter, text)
        if i == 0:
            scalebar(ax, 500)
    axp = fig.add_axes([0.14, 0.245, 0.82, 0.13])
    d = np.arange(c1 - c0)
    zp, zv = z["z_prior"][r0 + row, c0:c1], z["z_model"][r0 + row, c0:c1]
    axp.fill_between(d, zp, zv, where=zv < zp, color=OURS, alpha=0.25, lw=0, label="Lower in this model")
    axp.plot(d, zp, color=OTHER, lw=1.3, label="Copernicus")
    axp.plot(d, zv, color=OURS, lw=0.9, label="This model")
    axp.set_xlim(0, len(d))
    axp.set_xlabel("Distance along A–A′ (m)")
    axp.set_ylabel("Elevation (m)")
    axp.spines[["top", "right"]].set_visible(False)
    axp.legend(loc="upper right", fontsize=6.8, ncol=1, handlelength=1.4, labelspacing=0.25)
    axp.text(-0.16, 1.02, "d", transform=axp.transAxes, fontsize=12, fontweight="bold")
    caption(fig, f"Fig. {fig_no} | {place}.",
            "a, NAIP 2021 photograph (USDA). b, Copernicus resampled to 1 m (the faint grid is resampling "
            "ringing). c, This model. d, Elevation along A–A′; shading marks where this model is lower than "
            "Copernicus, which measures the forest canopy top.", notice=True)
    fig.savefig(out, dpi=DPI)


@styled
def region_map(overview: Path, out: Path, places=PLACES, box=FIGURE_PLACE[1:],
               fig_ref: str = "Figs. 1 & 3") -> None:
    ov = np.load(overview)
    sh = ov["shade"].astype(np.float32) / 255
    lid, sea = ov["lidar"], ov["sea"]
    x0, y0, x1, y1 = ov["bounds"]
    h, w = sh.shape
    land = ~sea
    rows, cols = np.where(land)
    r0, r1 = max(0, rows.min() - 20), min(h, rows.max() + 20)
    c0, c1 = max(0, cols.min() - 20), min(w, cols.max() + 20)

    def px(lon, lat):
        x = math.radians(lon) * EARTH_R
        y = EARTH_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        return (x - x0) / (x1 - x0) * w, (y1 - y) / (y1 - y0) * h

    rgb = np.dstack([sh * 0.80 + 0.17] * 3)
    recon = land & ~lid
    tint = np.array([0.33, 0.56, 0.82])
    rgb[recon] = rgb[recon] * 0.76 + tint * 0.24
    rgb[sea] = (1, 1, 1)
    fig = Figure(figsize=FIG, dpi=DPI)
    ax = fig.add_axes([0.13, 0.2, 0.83, 0.78])
    ax.imshow(rgb[r0:r1, c0:c1], extent=(c0, c1, r1, r0), interpolation="lanczos")
    ax.contour(np.arange(c0, c1), np.arange(r0, r1), (lid & land)[r0:r1, c0:c1].astype(float), levels=[0.5],
               colors="#222222", linewidths=0.55)
    lons, lats = [-156.25, -156.1, -155.95], [20.6, 20.7, 20.8, 20.9]
    ax.set_xticks([px(lo, 20.8)[0] for lo in lons])
    ax.set_xticklabels([f"{-lo:.2f}°W" for lo in lons])
    ax.set_yticks([px(-156.1, la)[1] for la in lats])
    ax.set_yticklabels([f"{la:.1f}°N" for la in lats])
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    for name, lon, lat, ha in places:
        X, Y = px(lon, lat)
        ax.plot(X, Y, "o", ms=2.6, color="black", mec="white", mew=0.6)
        ax.text(X + (6 if ha == "left" else -6), Y, name, fontsize=7.5, va="center", ha=ha, path_effects=HALO)
    lon, lat, size = box
    X, Y = px(lon, lat)
    hs = size / 2 / ((x1 - x0) / w) / math.cos(math.radians(lat))
    ax.add_patch(Rectangle((X - hs, Y - hs), 2 * hs, 2 * hs, fill=False, ec=TRANSECT, lw=1.1))
    ax.text(X + hs + 6, Y + hs, fig_ref, color=TRANSECT, fontsize=7.5, fontweight="bold", va="top",
            path_effects=HALO)
    LX, LY = px(-156.205, 20.835)
    ax.text(LX, LY, "≈490 km²\nreconstructed\n(no lidar)",
            fontsize=8, ha="center", va="center", color="#16344f", fontweight="bold", path_effects=HALO)
    scalebar(ax, 5000, w / ((x1 - x0) * math.cos(math.radians(20.8))), xf=0.05, yf=0.06)
    recon_fc = tuple(np.array([0.74] * 3) * 0.76 + tint * 0.24)
    ax.legend(handles=[Patch(fc=recon_fc, ec="none", label="Reconstructed (no lidar)"),
                       Patch(fc="#c4c4c4", ec="#222222", lw=0.5, label="Lidar, NOAA 2022 (training data)")],
              loc="lower right", fontsize=6.8, frameon=True, framealpha=0.92, edgecolor="none")
    caption(fig, "Fig. 2 | East Maui.",
            "Shaded relief of the published 1 m map. Blue: terrain reconstructed from imagery. Grey, black "
            f"outline: lidar coverage. Red box: {fig_ref}.", notice=True)
    fig.savefig(out, dpi=DPI)


@styled
def accuracy(out: Path, results: Path) -> None:
    groups, wins = accuracy_groups(results)
    fig = Figure(figsize=FIG, dpi=DPI)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 3], left=0.38, right=0.9, top=0.9, bottom=0.235,
                          hspace=1.05)
    for i, (title, n, xl, rows) in enumerate(groups):
        ax = fig.add_subplot(gs[i])
        names = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        ours = [not (nm.startswith("Copernicus") or nm == "FABDEM") for nm in names]
        ax.barh(range(len(vals)), vals, color=[OURS if o else OTHER for o in ours], height=0.62)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=7.8)
        for yy, (v, o) in enumerate(zip(vals, ours)):
            ax.text(v + max(vals) * 0.02, yy, f"{v:.2f}", va="center", fontsize=8,
                    fontweight="bold" if o else "normal")
            if o:
                ax.get_yticklabels()[yy].set_fontweight("bold")
        ax.set_xlim(0, max(vals) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xl, fontsize=7.2, labelpad=2)
        pos = ax.get_position()
        fig.text(0.04, pos.y1 + 0.052, title, fontsize=8.3, fontweight="bold")
        fig.text(0.04, pos.y1 + 0.024, n, fontsize=7, color="#555555")
    caption(fig, "Fig. 4 | Accuracy against independent ground truth.",
            "Median error, lower is better. a, Validation chips from the 2022 lidar survey (they also guided "
            f"model selection); FABDEM predates it. This model is better on {wins:.0%} of chips. "
            "b, ICESat-2 ground photons over the reconstructed area only, offset removed per overpass. "
            "c, Scored with a single model trained without this region.",
            y=0.012)
    fig.savefig(out, dpi=DPI)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True, type=Path, help="crop npz from figures.panels crop")
    ap.add_argument("--overview", required=True, type=Path, help="npz from figures.panels overview")
    ap.add_argument("--results", required=True, type=Path,
                    help=f"directory with {VAL}, {ICESAT} and {MOLOKAI}")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--place-name", default=PLACE_NAME)
    ap.add_argument("--places", type=Path, help="JSON list of [name, lon, lat, 'left'|'right'] map labels")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    places = json.loads(a.places.read_text(encoding="utf-8")) if a.places else PLACES
    before_after(a.panel, a.out / "p1_before_after.png", a.place_name)
    region_map(a.overview, a.out / "p2_east_maui.png", places)
    full_figure(a.panel, a.out / "p3_full_figure.png", a.place_name)
    accuracy(a.out / "p4_accuracy.png", a.results)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
