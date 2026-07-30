#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycirclize import Circos


DEFAULT_BASE_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw_celltocellspatial"
)


CELL_TYPE_COLORS = {
    "L2/3-IT": "#54990F",
    "L3/5-IT": "#279E68",
    "L6b": "#EEA1FD",
    "L5-IT": "#CFE6B8",
    "L5-ET(PT)": "#FB8AC6",
    "L6-IT": "#FF595E",
    "L6-CT": "#FF7291",
    "L6-IT(Car3)": "#7A81AC",
    "LAMP5": "#A3A6BC",
    "VIP": "#CCCCCC",
    "RELN": "#45B4D1",
    "PV_CHC": "#4CC9F0",
    "PV": "#FEB931",
    "SST": "#FEE440",
    "OPC": "#FB8B24",
    "Oli": "#BF9F47",
    "Ast": "#825F44",
    "Mic": "#6A0572",
    "VLMC/EC": "#3366CC",
    "L4-IT": "#008F7A",
}

CELLTYPE_RELABEL = {
    "L3/5-IT.03": "L4-IT.00",
    "L3/5-IT.04": "L4-IT.01",
    "L3/5-IT.05": "L4-IT.02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot subclass-level chord diagrams by layer.")
    parser.add_argument("--long-csv", default=str(DEFAULT_BASE_DIR / "niches_region_layer_radius100_celltocellspatial_long_nonzero.csv.gz"))
    parser.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR / "chord_by_layer"))
    parser.add_argument("--layers", nargs="*", default=["l1", "l2", "l3s", "l3d", "l5", "l6"])
    return parser.parse_args()


def subclass(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(".", n=1).str[0]


def relabel_celltypes(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(CELLTYPE_RELABEL)


def main() -> None:
    args = parse_args()
    long_csv = Path(args.long_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = {}
    rows = 0
    for chunk in pd.read_csv(
        long_csv,
        usecols=["layer.Sending", "SendingType", "ReceivingType", "score"],
        chunksize=500_000,
    ):
        rows += len(chunk)
        chunk["SendingType"] = relabel_celltypes(chunk["SendingType"])
        chunk["ReceivingType"] = relabel_celltypes(chunk["ReceivingType"])
        chunk["source_subclass"] = subclass(chunk["SendingType"])
        chunk["target_subclass"] = subclass(chunk["ReceivingType"])
        grouped = (
            chunk.groupby(["layer.Sending", "source_subclass", "target_subclass"], observed=True)["score"]
            .sum()
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            key = (row[0], row[1], row[2])
            agg[key] = agg.get(key, 0.0) + float(row[3])
        print(f"processed_rows {rows}", flush=True)

    agg_df = pd.DataFrame(
        [{"layer": k[0], "source_subclass": k[1], "target_subclass": k[2], "score_sum": v} for k, v in agg.items()]
    )
    agg_df.to_csv(out_dir / "celltocellspatial_subclass_score_matrix_by_layer.tsv", sep="\t", index=False)

    cell_order = list(CELL_TYPE_COLORS)
    all_cell_types = sorted(set(agg_df["source_subclass"]).union(set(agg_df["target_subclass"])))
    ordered_cell_types = [x for x in cell_order if x in all_cell_types] + [x for x in all_cell_types if x not in cell_order]
    fallback = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_map = dict(CELL_TYPE_COLORS)
    for idx, ct in enumerate([x for x in ordered_cell_types if x not in color_map]):
        color_map[ct] = fallback[idx % len(fallback)]

    def build_matrix(layer: str) -> pd.DataFrame:
        layer_df = agg_df[agg_df["layer"] == layer]
        present = sorted(set(layer_df["source_subclass"]).union(set(layer_df["target_subclass"])))
        types = [x for x in ordered_cell_types if x in present]
        matrix = pd.DataFrame(0.0, index=types, columns=types)
        for row in layer_df.itertuples(index=False):
            matrix.loc[row.source_subclass, row.target_subclass] += row.score_sum
        keep = (matrix.sum(axis=1) + matrix.sum(axis=0)) > 0
        return matrix.loc[keep, keep]

    def draw_chord(matrix: pd.DataFrame, layer: str):
        fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})
        # pycirclize can be sensitive to tiny floating point over-runs at sector
        # boundaries, so use integer weights for plotting while preserving ratios.
        plot_matrix = (matrix / matrix.to_numpy().sum() * 1_000_000).round().astype(int)
        keep = (plot_matrix.sum(axis=1) + plot_matrix.sum(axis=0)) > 0
        plot_matrix = plot_matrix.loc[keep, keep]
        cmap = {ct: color_map.get(ct, "#7f7f7f") for ct in plot_matrix.index}
        circos = Circos.chord_diagram(
            plot_matrix,
            space=3,
            cmap=cmap,
            r_lim=(93, 100),
            link_kws=dict(direction=1, ec="black", lw=0.15, alpha=0.75),
            label_kws=dict(r=106, size=10, color="black"),
        )
        circos.plotfig(ax=ax)
        ax.set_title(f"CellToCellSpatial chord diagram ({layer})", fontsize=18, pad=22)
        fig.tight_layout()
        return fig

    layers = [x for x in args.layers if x in set(agg_df["layer"])]
    layers.extend(sorted(set(agg_df["layer"]) - set(layers)))
    combined_pdf = out_dir / "celltocellspatial_chord_by_layer_all_layers.pdf"
    summary_lines = [f"long_csv {long_csv}", f"rows {rows}", f"combined_pdf {combined_pdf}"]

    with PdfPages(combined_pdf) as pdf:
        for layer in layers:
            matrix = build_matrix(layer)
            matrix_file = out_dir / f"celltocellspatial_chord_matrix_{layer}.csv"
            matrix.to_csv(matrix_file)
            fig = draw_chord(matrix, layer)
            pdf.savefig(fig, bbox_inches="tight")
            layer_pdf = out_dir / f"celltocellspatial_chord_{layer}.pdf"
            layer_png = out_dir / f"celltocellspatial_chord_{layer}.png"
            fig.savefig(layer_pdf, bbox_inches="tight")
            fig.savefig(layer_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            summary_lines.append(f"{layer}\tpdf={layer_pdf}\tpng={layer_png}\tmatrix={matrix_file}")

    summary_file = out_dir / "celltocellspatial_chord_by_layer_summary.txt"
    summary_file.write_text("\n".join(summary_lines) + "\n")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()
