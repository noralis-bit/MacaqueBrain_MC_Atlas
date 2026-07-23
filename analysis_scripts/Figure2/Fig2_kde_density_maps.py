from pathlib import Path
import re

import zhplot
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from correction_para import CorrectionPara
from ntp_manager import parcellate
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

CHIPS = ["T912", "T915", "T925"]
CELLTYPE = "L3_5-IT.05"
UM_PER_PIXEL = 0.5
GRID_N = 260
BANDWIDTHS = [500]
ARIAL_FONT = Path("/home/chaiqw/project/macaque_motor_new/fonts/arial.ttf")

font_manager.fontManager.addfont(ARIAL_FONT)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SOURCE_DIR = Path("/mnt/inner-data/ntp/macaque/Mq1234_all_motor")
CELL_DIR = Path("/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/trans_results/selected_L35_IT05_in_three_regions_warped_rxry")
OUT_DIR = Path("/home/chaiqw/project/macaque_motor_new/kde_density_maps_with_motor_outline_912_915_925")
F_LABEL_RE = re.compile(r"^[LR]-F([1-7])(?:[-#].*)?$")


def f_id(region) -> str | None:
    match = F_LABEL_RE.match(region.label.name)
    if match is None:
        return None
    return f"F{match.group(1)}"


def iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms
    elif isinstance(geom, GeometryCollection):
        for one in geom.geoms:
            yield from iter_polygons(one)


def chip_meta(chip: str):
    cp = CorrectionPara.select_one(chip)
    if cp is None:
        raise ValueError(f"missing CorrectionPara: {chip}")
    w, h = cp.w, cp.h
    is_first_monkey = int(chip[1:]) < 167
    if is_first_monkey:
        w, h = h, w
    return cp, w, h, is_first_monkey


def warp_points(points, chip: str) -> np.ndarray:
    cp, _, _, is_first_monkey = chip_meta(chip)
    arr = np.asarray(points)
    if is_first_monkey:
        arr = arr[:, [1, 0]]
    return cp.warp_point(arr, with_offset=False)


def warp_polygon(poly: Polygon, chip: str) -> Polygon:
    exterior = warp_points(np.array(poly.exterior.coords), chip)
    interiors = [warp_points(np.array(ring.coords), chip) for ring in poly.interiors]
    out = Polygon(exterior, interiors)
    if not out.is_valid:
        out = out.buffer(0)
    return out


def warp_geom(geom, chip: str):
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        return warp_polygon(geom, chip)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([warp_polygon(poly, chip) for poly in geom.geoms])
    if isinstance(geom, GeometryCollection):
        return GeometryCollection([warp_geom(one, chip) for one in geom.geoms])
    return geom


def parcellate_chip(path: Path, chip: str):
    _, w, h, _ = chip_meta(chip)
    return parcellate(path, um_per_pixel=UM_PER_PIXEL, bin_size=1, return_bytes=False, w=w, h=h)


def motor_mask(chip: str):
    sm = parcellate_chip(SOURCE_DIR / f"Mq277-{chip}-P0.ntp", chip)
    f_regions = [(f_id(region), warp_geom(region.polygon, chip)) for region in sm.regions if f_id(region) is not None]
    if chip == "T912":
        f3 = [poly for fid, poly in f_regions if fid == "F3"]
        f6 = [poly for fid, poly in f_regions if fid == "F6"]
        f1 = [poly for fid, poly in f_regions if fid == "F1"]
        f3_union = unary_union(f3)
        connected_f1 = [poly for poly in f1 if poly.buffer(1e-6).intersects(f3_union)]
        return unary_union(f3 + f6 + connected_f1)
    return unary_union([poly for _, poly in f_regions])


def load_cells(chip: str) -> pd.DataFrame:
    suffix = "F3+F6+connected_F1" if chip == "T912" else "all_F"
    return pd.read_parquet(CELL_DIR / f"{chip}_{CELLTYPE}_{suffix}.parquet")


def kde_grid(cells: pd.DataFrame, mask, bandwidth: float):
    xmin, ymin, xmax, ymax = mask.bounds
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04
    x = np.linspace(xmin - pad_x, xmax + pad_x, GRID_N)
    y = np.linspace(ymin - pad_y, ymax + pad_y, GRID_N)
    xx, yy = np.meshgrid(x, y)
    coords = cells[["spatial_rx", "spatial_ry"]].to_numpy()
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    density_flat = np.zeros(len(grid))
    kernel_norm = 1 / (2 * np.pi * bandwidth**2)
    for start in range(0, len(grid), 8_000):
        chunk = grid[start : start + 8_000]
        delta = chunk[:, None, :] - coords[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        density_flat[start : start + len(chunk)] = np.exp(-dist2 / (2 * bandwidth**2)).sum(axis=1) * kernel_norm
    density_per_px2 = density_flat.reshape(xx.shape)
    density_per_mm2 = density_per_px2 * (1_000_000 / (UM_PER_PIXEL**2))
    inside = np.array([mask.covers(Point(px, py)) for px, py in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
    density_per_mm2[~inside] = np.nan
    return xx, yy, density_per_mm2


def draw_outline(ax, mask, *, linewidth: float = 1.4) -> None:
    for poly in iter_polygons(mask):
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=linewidth, zorder=4)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, color="black", linewidth=linewidth * 0.8, zorder=4)


def plot_single(chip: str, mask, cells: pd.DataFrame, xx, yy, density, vmax: float, bandwidth: int) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    image = ax.pcolormesh(xx, yy, density, shading="auto", cmap="magma", vmin=0, vmax=vmax)
    levels = np.linspace(vmax * 0.2, vmax * 0.9, 5)
    ax.contour(xx, yy, np.nan_to_num(density, nan=0), levels=levels, colors="white", linewidths=0.45, alpha=0.75)
    ax.scatter(cells["spatial_rx"], cells["spatial_ry"], s=5, color="#36cfc9", alpha=0.55, linewidths=0, zorder=5)
    xmin, ymin, xmax, ymax = mask.bounds
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymax + pad_y, ymin - pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{chip} {CELLTYPE} KDE bw={bandwidth} n={len(cells)}")
    ax.set_axis_off()
    cbar = fig.colorbar(image, ax=ax, shrink=0.8)
    cbar.set_label("cells / mm^2")
    out = OUT_DIR / f"bw_{bandwidth}" / f"{chip}_{CELLTYPE}_kde_density_bw{bandwidth}_no_outline.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_combined(results, vmax: float, bandwidth: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    image = None
    for ax, item in zip(axes, results):
        chip = item["chip"]
        mask = item["mask"]
        cells = item["cells"]
        xx = item["xx"]
        yy = item["yy"]
        density = item["density"]
        image = ax.pcolormesh(xx, yy, density, shading="auto", cmap="magma", vmin=0, vmax=vmax)
        levels = np.linspace(vmax * 0.2, vmax * 0.9, 5)
        ax.contour(xx, yy, np.nan_to_num(density, nan=0), levels=levels, colors="white", linewidths=0.35, alpha=0.7)
        ax.scatter(cells["spatial_rx"], cells["spatial_ry"], s=3, color="#36cfc9", alpha=0.45, linewidths=0, zorder=5)
        xmin, ymin, xmax, ymax = mask.bounds
        pad_x = (xmax - xmin) * 0.05
        pad_y = (ymax - ymin) * 0.05
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymax + pad_y, ymin - pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{chip} bw={bandwidth} n={len(cells)}")
        ax.set_axis_off()
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("cells / mm^2")
    out = OUT_DIR / f"bw_{bandwidth}" / f"{CELLTYPE}_kde_density_bw{bandwidth}_no_outline_combined.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = [{"chip": chip, "mask": motor_mask(chip), "cells": load_cells(chip)} for chip in CHIPS]
    for bandwidth in BANDWIDTHS:
        results = []
        for item in base:
            xx, yy, density = kde_grid(item["cells"], item["mask"], bandwidth)
            results.append({**item, "xx": xx, "yy": yy, "density": density})
        vmax = float(max(np.nanpercentile(item["density"], 99) for item in results))
        for item in results:
            out = plot_single(item["chip"], item["mask"], item["cells"], item["xx"], item["yy"], item["density"], vmax, bandwidth)
            print(out)
        combined = plot_combined(results, vmax, bandwidth)
        print(combined)
        print("bandwidth", bandwidth, "vmax_99pct", vmax)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
