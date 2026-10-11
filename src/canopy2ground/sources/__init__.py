"""Readers for the public rasters, each returning a 1 m grid over projected bounds.

=================  ======================================  ==========================
module             data                                    role
=================  ======================================  ==========================
``pictometry``     Maui County Pictometry 2023, 0.28 m     base photograph (``rgb``)
``vivid``          Maxar Vivid 2022, 0.6 m (State GIS)     extra layer; rgb fallback
``naip``           NAIP 2021, 0.6 m R G B NIR              extra layer
``lidar``          NOAA 2022 bare-earth DTM, 0.5 m         training target
=================  ======================================  ==========================

HTTP sources go through one ``http.Fetcher`` and an optional ``cache``.
"""
from __future__ import annotations

import numpy as np

from . import lidar, naip, pictometry, vivid
from .cache import BlobCache, MemoryCache, SqliteCache
from .http import USER_AGENT, Fetcher

LAYERS = ("vivid", "naip")


def read_layer(name: str, bounds, crs: str, fetcher: Fetcher,
               cache: BlobCache | None = None) -> tuple[np.ndarray, float]:
    """Extra layer ``name`` as (H, W, C) uint8 and its coverage fraction (an all-zero image at coverage 0)."""
    if name == "vivid":
        arr, coverage = vivid.read(bounds, crs, fetcher, cache)
    elif name == "naip":
        arr, coverage = naip.read(bounds, crs)
    else:
        raise ValueError(f"unknown layer {name!r}; expected one of {LAYERS}")
    if coverage == 0:
        arr = np.zeros_like(arr)
    return np.moveaxis(np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8), 0, -1), coverage


__all__ = ["LAYERS", "USER_AGENT", "Fetcher", "MemoryCache", "SqliteCache", "lidar", "naip", "pictometry",
           "read_layer", "vivid"]
