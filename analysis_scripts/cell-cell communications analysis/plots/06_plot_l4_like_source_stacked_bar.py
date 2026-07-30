#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BASE_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw_celltocellspatial"
)
L4_LIKE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stacked counts for L4-like source cell types in l3s/l3d.")
    parser.add_argument("--long-csv", default=str(DEFAULT_BASE_DIR / "niches_region_layer_radius100_celltocellspatial_long_nonzero.csv.gz"))
    parser.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR / "barplots"))
    parser.add_argument("--celltypes", nargs="+", default=["L4-IT.00", "L4-IT.01", "L4-IT.02"])
    parser.add_argument("--layers", nargs="+", default=["l3s", "l3d"])
    parser.add_argument("--prefix", default="L4-like-source")
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

    count_data = pd.DataFrame(0, index=args.layers, columns=args.celltypes, dtype="int64")
    for chunk in pd.read_csv(args.long_csv, usecols=["layer.Sending", "SendingType"], chunksize=500_000):
        chunk["SendingType"] = relabel_celltypes(chunk["SendingType"])
        filtered = chunk[chunk["SendingType"].isin(args.celltypes) & chunk["layer.Sending"].isin(args.layers)]
        if filtered.empty:
            continue
        grouped = filtered.groupby(["layer.Sending", "SendingType"]).size()
        for (layer, celltype), n in grouped.items():
            count_data.loc[layer, celltype] += int(n)

    count_data.to_csv(out_dir / f"{args.prefix}_counts.csv", index_label="layer")

    fig, ax = plt.subplots(figsize=(4, 4))
    count_data.plot(kind="bar", stacked=True, ax=ax, width=0.6, edgecolor="none", color=L4_LIKE_COLORS[: len(args.celltypes)])
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("N Interacts", fontsize=12, fontweight="bold")
    ax.set_xticklabels(args.layers, rotation=0, fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.prefix}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{args.prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("pdf", out_dir / f"{args.prefix}.pdf")
    print("png", out_dir / f"{args.prefix}.png")
    print("csv", out_dir / f"{args.prefix}_counts.csv")
    print(count_data)


if __name__ == "__main__":
    main()
