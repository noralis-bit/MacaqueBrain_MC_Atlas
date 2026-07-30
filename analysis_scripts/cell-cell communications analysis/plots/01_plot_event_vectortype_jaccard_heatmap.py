#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_BASE_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw_celltocellspatial"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot region-layer Jaccard heatmap using event + VectorType as the event unit."
    )
    parser.add_argument("--long-csv", default=str(DEFAULT_BASE_DIR / "niches_region_layer_radius100_celltocellspatial_long_nonzero.csv.gz"))
    parser.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR / "jaccard_heatmap_event_vectortype"))
    parser.add_argument("--prefix", default="event_vectortype")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    long_csv = Path(args.long_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    region_priority = ["F6", "F3", "F7", "F5", "F1", "F2", "F4"]
    layer_priority = ["l1", "l2", "l3s", "l3d", "l5", "l6"]

    sets_by_region: dict[str, set[str]] = {}
    rows = 0
    for chunk in pd.read_csv(
        long_csv,
        usecols=["region_layer_result", "event", "VectorType"],
        chunksize=500_000,
    ):
        rows += len(chunk)
        chunk["event_VectorType"] = chunk["event"].astype(str) + "|" + chunk["VectorType"].astype(str)
        for label, sub in chunk.groupby("region_layer_result", sort=False):
            sets_by_region.setdefault(str(label), set()).update(sub["event_VectorType"].unique())
        print(f"processed_rows {rows}", flush=True)

    all_labels = sorted(sets_by_region)
    custom_order = []
    for layer in layer_priority:
        for region in region_priority:
            label = f"{region}-{layer}"
            if label in sets_by_region:
                custom_order.append(label)
    custom_order.extend([label for label in all_labels if label not in custom_order])

    n = len(custom_order)
    jaccard = np.eye(n, dtype=float)
    for i, label_i in enumerate(custom_order):
        a = sets_by_region[label_i]
        for j in range(i + 1, n):
            b = sets_by_region[custom_order[j]]
            union = len(a | b)
            sim = np.nan if union == 0 else len(a & b) / union
            jaccard[i, j] = sim
            jaccard[j, i] = sim

    jaccard_df = pd.DataFrame(jaccard, index=custom_order, columns=custom_order)
    jaccard_csv = out_dir / f"{args.prefix}_jaccard_similarity_matrix.csv"
    jaccard_df.to_csv(jaccard_csv, index_label="region_layer")
    (out_dir / f"{args.prefix}_jaccard_order.txt").write_text("\n".join(custom_order) + "\n")

    colors = [
        (0.0, "#2166ac"),
        (0.05, "#2166ab"),
        (0.15, "#ffffff"),
        (0.24, "#CD5C5C"),
        (1.0, "#B22222"),
    ]
    custom_cmap = LinearSegmentedColormap.from_list("user_exact_event_vectortype", colors)

    png_out = out_dir / f"{args.prefix}_jaccard_heatmap_exact_colors.png"
    pdf_out = out_dir / f"{args.prefix}_jaccard_heatmap_exact_colors.pdf"
    summary_out = out_dir / f"{args.prefix}_jaccard_heatmap_exact_colors.summary.txt"

    sns.set_theme(style="white", font_scale=0.72)
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        jaccard_df,
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        linewidths=0,
        annot=False,
        square=True,
        cbar_kws={
            "label": "Jaccard Similarity Coefficient",
            "shrink": 0.6,
            "aspect": 10,
            "pad": 0.04,
            "ticks": [0, 0.05, 0.15, 0.24, 0.5, 0.75, 1.0],
        },
        ax=ax,
    )
    ax.set_title("Jaccard Similarity of NICHES Cell-Cell Events: LR + VectorType", fontsize=16, pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    upper = jaccard_df.where(np.triu(np.ones(jaccard_df.shape, dtype=bool), k=1)).stack()
    n_events = len(set().union(*sets_by_region.values()))
    summary = "\n".join(
        [
            f"long_csv {long_csv}",
            f"rows {rows}",
            f"n_regions {n}",
            f"n_event_vectortype_total {n_events}",
            f"min_offdiag {upper.min():.6f}",
            f"median_offdiag {upper.median():.6f}",
            f"max_offdiag {upper.max():.6f}",
            "color_stops " + ";".join(f"{pos}:{color}" for pos, color in colors),
            "vmin 0",
            "vmax 1",
            "white_point 0.15",
            f"png {png_out}",
            f"pdf {pdf_out}",
            f"jaccard_csv {jaccard_csv}",
        ]
    )
    summary_out.write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
