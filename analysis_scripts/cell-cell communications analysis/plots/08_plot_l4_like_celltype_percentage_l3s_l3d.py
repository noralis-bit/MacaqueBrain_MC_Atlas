#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_H5AD = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/44.cross_main_region_cellchat/"
    "adata286.h5ad"
)
DEFAULT_BASE_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw_celltocellspatial"
)
L4_LIKE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot L4-like cell abundance percentages among all cells in l3s/l3d.")
    parser.add_argument("--h5ad", default=str(DEFAULT_H5AD))
    parser.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR / "barplots"))
    parser.add_argument("--celltypes", nargs="+", default=["L4-IT.00", "L4-IT.01", "L4-IT.02"])
    parser.add_argument("--layers", nargs="+", default=["l3s", "l3d"])
    parser.add_argument("--annotation-col", default="annotation")
    parser.add_argument("--layer-col", default="layer")
    parser.add_argument("--prefix", default="L4-like-percentage-of-all-cells")
    return parser.parse_args()


CELLTYPE_RELABEL = {
    "L3/5-IT.03": "L4-IT.00",
    "L3/5-IT.04": "L4-IT.01",
    "L3/5-IT.05": "L4-IT.02",
}


def relabel_celltypes(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(CELLTYPE_RELABEL)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs[[args.annotation_col, args.layer_col]].copy()
    obs[args.annotation_col] = relabel_celltypes(obs[args.annotation_col])
    obs[args.layer_col] = obs[args.layer_col].astype(str)

    layer_totals = obs[args.layer_col].value_counts().reindex(args.layers).fillna(0).astype(int)
    counts = pd.crosstab(obs[args.layer_col], obs[args.annotation_col]).reindex(
        index=args.layers,
        columns=args.celltypes,
        fill_value=0,
    )
    percentages = counts.div(layer_totals.replace(0, pd.NA), axis=0).fillna(0) * 100

    summary = percentages.copy()
    summary.insert(0, "all_cells_in_layer", layer_totals)
    for celltype in reversed(args.celltypes):
        summary.insert(1, f"{celltype}_n_cells", counts[celltype])
    summary.to_csv(out_dir / f"{args.prefix}.csv", index_label="layer")

    fig, ax = plt.subplots(figsize=(4, 4))
    percentages.plot(kind="bar", stacked=True, ax=ax, width=0.6, edgecolor="none", color=L4_LIKE_COLORS[: len(args.celltypes)])
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of all cells (%)", fontsize=12, fontweight="bold")
    ax.set_xticklabels(args.layers, rotation=0, fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(1.0, percentages.sum(axis=1).max() * 1.05))
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.prefix}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{args.prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("pdf", out_dir / f"{args.prefix}.pdf")
    print("png", out_dir / f"{args.prefix}.png")
    print("csv", out_dir / f"{args.prefix}.csv")
    print("counts")
    print(counts)
    print("layer_totals")
    print(layer_totals)
    print("percentages")
    print(percentages)


if __name__ == "__main__":
    main()
