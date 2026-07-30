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
import seaborn as sns


DEFAULT_BASE_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw_celltocellspatial"
)
LAYER_ORDER = ["l1", "l2", "l3s", "l3d", "l5", "l6"]
PALETTE = {"l1": "#82584d", "l2": "#8566a4", "l3s": "#519746", "l3d": "#e0833b", "l5": "#b93a32", "l6": "#3970a5"}
CELLTYPE_RELABEL = {"L3/5-IT.03": "L4-IT.00", "L3/5-IT.04": "L4-IT.01", "L3/5-IT.05": "L4-IT.02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot number of nonzero interaction records by layer for cellpairs.")
    parser.add_argument("--cellpairs", nargs="+", default=["Ast.00->L2/3-IT.04", "Ast.00->L4-IT.02"])
    parser.add_argument("--long-csv", default=str(DEFAULT_BASE_DIR / "niches_region_layer_radius100_celltocellspatial_long_nonzero.csv.gz"))
    parser.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR / "barplots"))
    return parser.parse_args()


def safe_name(cellpair: str) -> str:
    return cellpair.replace("/", "_").replace("->", "_to_").replace("|", "_")


def relabel_celltypes(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(CELLTYPE_RELABEL)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {pair: {layer: 0 for layer in LAYER_ORDER} for pair in args.cellpairs}

    rows_seen = 0
    for chunk in pd.read_csv(args.long_csv, usecols=["SendingType", "ReceivingType", "layer.Sending"], chunksize=500_000):
        rows_seen += len(chunk)
        chunk["SendingType"] = relabel_celltypes(chunk["SendingType"])
        chunk["ReceivingType"] = relabel_celltypes(chunk["ReceivingType"])
        chunk["cellpair"] = chunk["SendingType"].astype(str) + "->" + chunk["ReceivingType"].astype(str)
        filtered = chunk[chunk["cellpair"].isin(args.cellpairs)]
        if filtered.empty:
            continue
        grouped = filtered.groupby(["cellpair", "layer.Sending"], observed=True).size()
        for (pair, layer), n in grouped.items():
            if layer in counts[pair]:
                counts[pair][layer] += int(n)

    summary_lines = [f"rows_seen {rows_seen}"]
    for pair in args.cellpairs:
        count_stats = pd.DataFrame({"layer": LAYER_ORDER, "count": [counts[pair][layer] for layer in LAYER_ORDER]})
        count_stats.to_csv(out_dir / f"{safe_name(pair)}_interaction_count_by_layer.csv", index=False)
        count_stats_nonzero = count_stats[count_stats["count"] > 0].copy()

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(
            data=count_stats_nonzero,
            x="layer",
            y="count",
            hue="layer",
            order=LAYER_ORDER,
            hue_order=LAYER_ORDER,
            palette=PALETTE,
            dodge=False,
            legend=False,
            ax=ax,
        )
        for container in ax.containers:
            for patch in container.patches:
                height = patch.get_height()
                if height > 0:
                    ax.text(patch.get_x() + patch.get_width() / 2, height + 1, str(int(height)), ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xlabel("Group", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Interactions", fontsize=12, fontweight="bold")
        ax.set_title(pair, fontsize=14, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if count_stats["count"].max() > 0:
            ax.set_ylim(0, count_stats["count"].max() * 1.15)
        fig.tight_layout()
        pdf = out_dir / f"{safe_name(pair)}_interaction_count_by_layer.pdf"
        png = out_dir / f"{safe_name(pair)}_interaction_count_by_layer.png"
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        summary_lines.append(f"{pair}\tpdf={pdf}\tpng={png}")

    summary_file = out_dir / "cellpair_interaction_count_by_layer_summary.txt"
    summary_file.write_text("\n".join(summary_lines) + "\n")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()
