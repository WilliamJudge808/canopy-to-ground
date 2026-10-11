import hashlib

import pytest

from canopy2ground import grid


def test_chip_id_round_trip_and_bounds():
    cid = grid.chip_id(1544, 4489)
    assert cid == "maui-+01544-+04489"
    assert grid.parse_chip_id(cid) == (1544, 4489)
    assert grid.chip_bounds(cid) == (1544 * 512.0, 4489 * 512.0, 1545 * 512.0, 4490 * 512.0)


def test_chip_index_is_floor_division_and_neighbours_tile_without_gaps():
    b = grid.chip_bounds(grid.chip_id(1544, 4489))
    assert grid.chip_index(b.xmin, b.ymin) == (1544, 4489)
    assert grid.chip_index(b.xmax - 1e-6, b.ymax - 1e-6) == (1544, 4489)
    assert grid.chip_index(b.xmax, b.ymin) == (1545, 4489)
    east = grid.chip_bounds(grid.chip_id(1545, 4489))
    north = grid.chip_bounds(grid.chip_id(1544, 4490))
    assert east.xmin == b.xmax and north.ymin == b.ymax


def test_parse_rejects_other_prefixes():
    with pytest.raises(ValueError):
        grid.parse_chip_id("oahu-+01000-+02000")


def test_block_split_is_the_md5_rule():
    for bx, by in [(0, 0), (77, 224), (65, 230), (3, -2)]:
        h = int(hashlib.md5(f"maui_{bx}_{by}".encode()).hexdigest(), 16)
        assert grid.block_split(bx, by) == ("val" if h % 100 < 20 else "train")
    splits = [grid.block_split(x, y) for x in range(100) for y in range(100)]
    assert 0.18 < splits.count("val") / len(splits) < 0.22


def test_known_cache_chips_fall_in_val_blocks():
    for cid in ("maui-+01544-+04489", "maui-+01398-+04488", "maui-+01336-+04566", "maui-+01455-+04543"):
        assert grid.chip_split(*grid.parse_chip_id(cid)) == "val"


def test_buffer_is_the_one_chip_ring_around_val_blocks():
    # block (77, 224) is val, (77, 223) and (76, 223) are train
    assert grid.block_split(77, 224) == "val" and grid.block_split(77, 223) == "train"
    assert grid.in_buffer(1540, 4479)                  # touches the val block to the north
    assert not grid.in_buffer(1540, 4478)              # two chips away
    assert grid.in_buffer(1540, 4478, ring=2)
    assert not grid.in_buffer(1544, 4489)              # val chips are never buffer


def test_corpus_box_covers_maui_county():
    pytest.importorskip("pyproj")
    cells = grid.chips_in_lonlat(*grid.CORPUS_LONLAT)
    assert len(cells) == len(set(cells))
    xs, ys = zip(*cells)
    assert min(xs) <= 1336 and max(xs) >= 1544 and min(ys) <= 4488 and max(ys) >= 4566
    assert len(cells) == (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
