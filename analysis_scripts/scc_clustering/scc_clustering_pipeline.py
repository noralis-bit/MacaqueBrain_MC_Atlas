#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCC (Spatial Constrained Clustering) analysis pipeline for motor cortex
spatial transcriptomics data.

This script consolidates the SCC clustering workflow used in the manuscript:
  1. Build a neighbour-count AnnData from cell-bin annotations (CELLannot).
  2. Per-slide preprocessing (PCA, KNN).
  3. Spateo SCC clustering at multiple resolutions (louvain).
  4. Map SCC clusters to cortical layers and visualise the result with
     slice outline overlays.

The script is designed for paper code sharing: every path / parameter that
is dataset specific is exposed as a CLI argument or a constant at the top of
the file, so reviewers can re-run it on their own data with minimal edits.

Author: Ding Lingjun
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from functools import partial
from multiprocessing import Pool

import anndata as ad
import dynamo as dyn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spateo as st
from sklearn.neighbors import KDTree

# ---------------------------------------------------------------------------
# Constants / defaults (override via CLI or edit here)
# ---------------------------------------------------------------------------
DEFAULT_TRANSFER_DIR = "/sdd/datasets/cellpaper_spatial_h5ad/mq286_anno_from_cellpaper"
DEFAULT_OUTLINE_DIR = "/sdd/datasets/motor_spatial_cellbins/mq286_outline"
DEFAULT_OUTPUT_DIR = "./scc_output"

# Motor regions used to filter slides that contain enough motor cortex
MOTOR_REGIONS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
MIN_MOTOR_RATIO = 0.1

# Neighbour matrix parameters
NEIGHBOR_RADIUS_UM = 150.0       # radius for the "surrounding" neighbourhood
PIXEL_PER_UM = 0.5               # stereo-seq bin pixel size in um
NEIGHBOR_THRESHOLD = NEIGHBOR_RADIUS_UM / PIXEL_PER_UM

# SCC parameters
N_PCA = 30
N_NEIGHBORS = 50
E_NEIGH = 80
RESOLUTIONS = [0.4, 0.5, 0.6, 0.7]

# Cortical-layer colour palette used for the final figure
LAYER_PALETTE = {
    "l1": "#8a564b",
    "l2": "#8b64a7",
    "l3a": "#359939",
    "l3b": "#ef7c21",
    "l4": "#e377c2",
    "l5": "#ca2728",
    "l6": "#1e70a8",
    "wm": "#9edae5",
    "other": "#c7c7c7",
}


# ---------------------------------------------------------------------------
# Step 1 - build neighbour-count AnnData from CELLannot transfer results
# ---------------------------------------------------------------------------
def build_neighbor_adata(
    transfer_dir: str,
    output_h5ad: str,
    motor_regions: list[str] | None = None,
    min_motor_ratio: float = MIN_MOTOR_RATIO,
    neighbor_threshold: float = NEIGHBOR_THRESHOLD,
) -> ad.AnnData:
    """Read cell-bin annotation parquets and build a neighbour-count AnnData.

    Each cell-bin is represented by the counts of surrounding cell types
    within ``neighbor_threshold`` pixels (KDTree radius query).
    """
    motor_regions = motor_regions or MOTOR_REGIONS

    transfer_files = glob.glob(os.path.join(transfer_dir, "*"))
    transfer_result = pd.concat(
        [pd.read_parquet(p) for p in transfer_files], axis=0, ignore_index=True
    )

    cellmeta_df = transfer_result[
        ["gene_area", "rx", "ry", "chip", "celltype", "SubClass", "Class"]
    ].copy()
    cellmeta_df["region"] = cellmeta_df["gene_area"].str.split("-").str[1]

    # keep only slides with enough motor cortex
    keep = []
    for chip in cellmeta_df.chip.unique():
        slide = cellmeta_df[cellmeta_df.chip == chip]
        ratio = slide.region.isin(motor_regions).mean()
        if ratio > min_motor_ratio:
            keep.append(slide)
    cellmeta_df = pd.concat(keep, axis=0, ignore_index=True)

    # flatten all slides onto a single coordinate system (identity translation
    # here; replace with real translations if you need a montage)
    cellmeta_df["x"] = cellmeta_df["rx"].astype(float)
    cellmeta_df["y"] = cellmeta_df["ry"].astype(float)

    # build neighbour count matrix
    tree = KDTree(cellmeta_df[["x", "y"]].values)
    neighbor_data = []
    for point in cellmeta_df[["x", "y"]].values:
        sur_indices = tree.query_radius([point], r=neighbor_threshold)
        sur_count = cellmeta_df.iloc[np.concatenate(sur_indices)][
            "celltype"
        ].value_counts()
        neighbor_data.append(sur_count)
    neighbor_df = pd.DataFrame(neighbor_data).fillna(0)

    neighbor_df.reset_index(drop=True, inplace=True)
    cellmeta_df.reset_index(drop=True, inplace=True)

    adata = ad.AnnData(X=neighbor_df.values, obs=cellmeta_df)
    adata.var_names = neighbor_df.columns.astype(str)
    adata.X = np.nan_to_num(adata.X, nan=0)

    adata.write_h5ad(output_h5ad)
    print(f"[build_neighbor_adata] wrote {output_h5ad}  ({adata.shape})")
    return adata


# ---------------------------------------------------------------------------
# Step 2 + 3 - per-slide SCC clustering
# ---------------------------------------------------------------------------
def scc_one_slide(slide: str, adata: ad.AnnData, out_dir: str) -> str:
    """Run PCA -> KNN -> SCC (multiple resolutions) on a single slide."""
    adata_slide = adata[adata.obs.chip == slide].copy()

    st.tl.pca_spateo(adata_slide, n_pca_components=N_PCA)
    dyn.tl.neighbors(adata_slide, n_neighbors=N_NEIGHBORS, cores=30)

    for res in RESOLUTIONS:
        key = f"scc_res{int(res * 10)}"
        st.tl.scc(
            adata_slide,
            spatial_key="spatial",
            resolution=res,
            e_neigh=E_NEIGH,
            cluster_method="louvain",
            key_added=key,
            pca_key="pca",
        )

    out_path = os.path.join(out_dir, f"scc_{slide}.h5ad")
    adata_slide.write_h5ad(out_path)
    print(f"[scc_one_slide] {slide} -> {out_path}")
    return out_path


def run_scc_all_slides(
    adata: ad.AnnData, out_dir: str, n_cores: int | None = None
) -> list[str]:
    """Run SCC on every slide in parallel."""
    os.makedirs(out_dir, exist_ok=True)
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype="f")
    adata.uns["__type"] = "UMI"
    adata.uns["pp"] = {"norm_method": None}

    slides = adata.obs.chip.unique().tolist()
    if n_cores and n_cores > 1:
        with Pool(processes=n_cores) as pool:
            paths = pool.map(partial(scc_one_slide, adata=adata, out_dir=out_dir), slides)
    else:
        paths = [scc_one_slide(s, adata, out_dir) for s in slides]
    return paths


# ---------------------------------------------------------------------------
# Step 4 - geometry helpers for outline overlay
# ---------------------------------------------------------------------------
def apply_correction_with_matrix(points, corr_para):
    """Apply a 3x3 transformation matrix stored in ``corr_para`` to 2D points."""
    points = np.asarray(points)
    matrix = np.array(corr_para["transformation_matrix"]).reshape(3, 3)
    homo = np.hstack([points, np.ones((len(points), 1))])
    return (homo @ matrix.T)[:, :2]


def apply_transformation_to_coords(coords, transformation_matrix):
    coords = np.asarray(coords)
    homo = np.hstack([coords, np.ones((coords.shape[0], 1))])
    matrix = np.array(transformation_matrix).reshape(3, 3)
    return (homo @ matrix.T)[:, :2]


def load_outline(ax, outline_dir: str, chip: str):
    """Draw the corrected slice outline onto ``ax`` and return the transform."""
    json_path = os.path.join(outline_dir, f"MQ286-{chip}-P0.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    for line in data["lines"]:
        pts = np.array(line["countours"])
        pts = apply_correction_with_matrix(pts, data["corr_para"])
        ax.plot(pts[:, 0], pts[:, 1], linewidth=0.5, color="grey", alpha=1)
    ax.set_aspect("equal")
    return data["corr_para"]["transformation_matrix"]


# ---------------------------------------------------------------------------
# Step 5 - final plotting
# ---------------------------------------------------------------------------
def plot_scc_layer(
    h5ad_path: str,
    outline_dir: str,
    map_dict: dict[str, str],
    out_figure: str | None = None,
    point_size: int = 7,
):
    """Read a per-slide SCC h5ad, map clusters -> layers, and plot."""
    adata = sc.read_h5ad(h5ad_path)
    adata.obs["region"] = adata.obs["gene_area"].str.split("-").str[1]
    adata.obs["area"] = adata.obs["gene_area"].str.split("-").str[2]

    adata.obs["scc_layer"] = (
        adata.obs[[k for k in map_dict if k in adata.obs][0]].astype(str).replace(map_dict)
    )
    adata.obs["scc_layer"] = adata.obs["scc_layer"].astype("category")

    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()

    matrix = load_outline(ax, outline_dir, adata.obs.chip.iloc[0])
    coords = adata.obs[["rx", "ry"]].values
    transformed = apply_transformation_to_coords(coords, matrix)
    adata.obs["x"] = transformed[:, 0]
    adata.obs["y"] = transformed[:, 1]
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype="f")

    sc.pl.embedding(
        adata,
        basis="spatial",
        color="scc_layer",
        palette=LAYER_PALETTE,
        size=point_size,
        alpha=1,
        ax=ax,
        show=False,
    )
    if out_figure:
        fig.savefig(out_figure, dpi=300, bbox_inches="tight")
        print(f"[plot_scc_layer] saved {out_figure}")
    return fig, ax


def plot_region_overlay(h5ad_path: str, outline_dir: str, out_figure: str | None = None):
    """Plot SCC raw clusters with region-name labels at centroids."""
    adata = sc.read_h5ad(h5ad_path)
    adata.obs["region"] = adata.obs["gene_area"].str.split("-").str[1]

    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()
    matrix = load_outline(ax, outline_dir, adata.obs.chip.iloc[0])
    coords = adata.obs[["rx", "ry"]].values
    transformed = apply_transformation_to_coords(coords, matrix)
    adata.obs["x"] = transformed[:, 0]
    adata.obs["y"] = transformed[:, 1]

    sc.pl.scatter(
        adata, alpha=1, x="x", y="y", color="region",
        color_map="Tab20", show=False, size=10, ax=ax,
    )
    for region in adata.obs["region"].unique():
        mask = adata.obs["region"] == region
        ax.text(
            adata.obs.loc[mask, "x"].mean(),
            adata.obs.loc[mask, "y"].mean(),
            str(region), fontsize=18, weight="bold",
            color="black", va="center", ha="center",
        )
    if out_figure:
        fig.savefig(out_figure, dpi=300, bbox_inches="tight")
        print(f"[plot_region_overlay] saved {out_figure}")
    return fig, ax


def plot_layer_composition_bar(h5ad_path: str, out_figure: str | None = None):
    """Stacked bar of layer composition per region."""
    adata = sc.read_h5ad(h5ad_path)
    if "scc_layer" not in adata.obs:
        raise ValueError("run plot_scc_layer first to add 'scc_layer' column")
    df = adata.obs.copy()
    count_df = df.groupby(["region", "scc_layer"]).size().unstack(fill_value=0)
    pct_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    pct_df.plot(kind="bar", stacked=True, ax=ax, color=[LAYER_PALETTE.get(c, "#999") for c in pct_df.columns])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Layer composition per region")
    ax.legend(title="Layer")
    plt.tight_layout()
    if out_figure:
        fig.savefig(out_figure, dpi=300, bbox_inches="tight")
        print(f"[plot_layer_composition_bar] saved {out_figure}")
    return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SCC clustering pipeline for motor cortex spatial data")
    parser.add_argument("--transfer-dir", default=DEFAULT_TRANSFER_DIR,
                        help="dir with CELLannot parquet files")
    parser.add_argument("--outline-dir", default=DEFAULT_OUTLINE_DIR,
                        help="dir with MQ286-<chip>-P0.json outline files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="where to write h5ad / figures")
    parser.add_argument("--neighbor-h5ad", default=None,
                        help="skip neighbour build if this h5ad already exists")
    parser.add_argument("--n-cores", type=int, default=1,
                        help="parallel workers for per-slide SCC")
    parser.add_argument("--chip", default=None,
                        help="single chip to plot (e.g. T578). If omitted, plot all.")
    parser.add_argument("--map-dict", default=None,
                        help='JSON file with {"scc_cluster_id":"layer"} mapping')
    parser.add_argument("--scc-key", default="scc_res5",
                        help="which scc column to use for layer mapping")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scc_dir = os.path.join(args.output_dir, "scc")
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Step 1 - neighbour AnnData
    neighbor_h5ad = args.neighbor_h5ad or os.path.join(args.output_dir, "neighbor_adata.h5ad")
    if os.path.exists(neighbor_h5ad):
        print(f"Loading existing neighbour AnnData: {neighbor_h5ad}")
        adata = sc.read_h5ad(neighbor_h5ad)
    else:
        adata = build_neighbor_adata(args.transfer_dir, neighbor_h5ad)

    # Step 2+3 - SCC per slide
    scc_paths = run_scc_all_slides(adata, scc_dir, n_cores=args.n_cores)

    # Step 4+5 - plotting
    map_dict = None
    if args.map_dict and os.path.exists(args.map_dict):
        with open(args.map_dict) as f:
            map_dict = json.load(f)

    chips = [args.chip] if args.chip else None
    for p in scc_paths:
        chip = os.path.basename(p).replace("scc_", "").replace(".h5ad", "")
        if chips and chip not in chips:
            continue
        # raw SCC clusters
        plot_region_overlay(p, args.outline_dir,
                            os.path.join(fig_dir, f"{chip}_region_overlay.png"))
        if map_dict:
            plot_scc_layer(p, args.outline_dir, map_dict,
                           os.path.join(fig_dir, f"{chip}_scc_layer.png"))
            plot_layer_composition_bar(p,
                                       os.path.join(fig_dir, f"{chip}_composition.png"))


if __name__ == "__main__":
    main()
