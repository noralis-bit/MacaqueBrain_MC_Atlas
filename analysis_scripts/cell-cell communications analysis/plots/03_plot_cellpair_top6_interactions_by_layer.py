#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BASE_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw_celltocellspatial"
)
LAYER_ORDER = ["l6", "l5", "l3d", "l3s", "l2", "l1"]
PALETTE = {"l1": "#82584d", "l2": "#8566a4", "l3s": "#519746", "l3d": "#e0833b", "l5": "#b93a32", "l6": "#3970a5"}
CELLTYPE_RELABEL = {"L3/5-IT.03": "L4-IT.00", "L3/5-IT.04": "L4-IT.01", "L3/5-IT.05": "L4-IT.02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top 6 LR interactions by layer for one directed cellpair.")
    parser.add_argument("--cellpair", default=None, help="One directed cellpair, e.g. Ast.00->L4-IT.02.")
    parser.add_argument(
        "--cellpairs",
        nargs="+",
        default=None,
        help="One or more directed cellpairs. Overrides --cellpair when provided.",
    )
    parser.add_argument("--long-csv", default=str(DEFAULT_BASE_DIR / "niches_region_layer_radius100_celltocellspatial_long_nonzero.csv.gz"))
    parser.add_argument("--out-dir", default=str(DEFAULT_BASE_DIR / "barplots"))
    return parser.parse_args()


def safe_name(cellpair: str) -> str:
    return cellpair.replace("/", "_").replace("->", "_to_").replace("|", "_")


def relabel_celltypes(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(CELLTYPE_RELABEL)


def resolve_cellpairs(args: argparse.Namespace) -> list[str]:
    if args.cellpairs:
        return args.cellpairs
    if args.cellpair:
        return [args.cellpair]
    return ["Ast.00->L2/3-IT.04", "Ast.00->L4-IT.02"]


def add_scores(target: dict[tuple[str, str], float], grouped: pd.DataFrame) -> None:
    for event, layer, score in grouped.itertuples(index=False, name=None):
        target[(event, layer)] = target.get((event, layer), 0.0) + float(score)


def plot_one_cellpair(cellpair: str, agg: dict[tuple[str, str], float], out_dir: Path) -> Path:
    if not agg:
        raise SystemExit(f"No records found for {cellpair}")

    data = pd.DataFrame([{"interaction": k[0], "layer": k[1], "score": v} for k, v in agg.items()])
    order = data.groupby("interaction")["score"].sum().sort_values(ascending=False).index[:6]
    selected = data[data["interaction"].isin(order)].copy()

    stem = f"{safe_name(cellpair)}_top6_interactions_by_layer"
    selected.to_csv(out_dir / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 7))
    y_positions = []
    y_base = 0.0
    for interaction in order:
        sub = selected[selected["interaction"] == interaction]
        sub = sub.set_index("layer").reindex(LAYER_ORDER).dropna(subset=["score"]).reset_index()
        start_y = y_base
        for i, row in enumerate(sub.itertuples(index=False)):
            ax.barh(start_y + i * 0.5, row.score, height=0.5, color=PALETTE[row.layer], edgecolor=None, linewidth=0)
        y_positions.append(start_y + (len(sub) - 1) * 0.25 if len(sub) else start_y)
        y_base += len(sub) * 0.5 + 0.1

    ax.set_yticks(y_positions)
    ax.set_yticklabels(order, fontsize=12)
    ax.set_xlabel("Total Probability Strength", fontsize=12, fontweight="bold")
    ax.set_ylabel("Interaction", fontsize=12, fontweight="bold")
    ax.set_title(cellpair, fontsize=13, fontweight="bold", pad=15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[layer]) for layer in PALETTE]
    ax.legend(handles, PALETTE.keys(), title="Layer", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf


def main() -> None:
    args = parse_args()
    cellpairs = resolve_cellpairs(args)
    parsed_pairs = {pair: tuple(pair.split("->", 1)) for pair in cellpairs}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggs = {pair: {} for pair in cellpairs}
    rows_seen = 0
    rows_kept = {pair: 0 for pair in cellpairs}
    for chunk in pd.read_csv(
        args.long_csv,
        usecols=["SendingType", "ReceivingType", "event", "layer.Sending", "score"],
        chunksize=500_000,
    ):
        rows_seen += len(chunk)
        chunk["SendingType"] = relabel_celltypes(chunk["SendingType"])
        chunk["ReceivingType"] = relabel_celltypes(chunk["ReceivingType"])
        for pair, (sender, receiver) in parsed_pairs.items():
            selected = chunk[(chunk["SendingType"] == sender) & (chunk["ReceivingType"] == receiver)]
            rows_kept[pair] += len(selected)
            if selected.empty:
                continue
            grouped = selected.groupby(["event", "layer.Sending"], observed=True)["score"].sum().reset_index()
            add_scores(aggs[pair], grouped)
        print("processed_rows=" + str(rows_seen) + " " + " ".join(f"{pair}={rows_kept[pair]}" for pair in cellpairs), flush=True)

    print(f"rows_seen={rows_seen}")
    for pair in cellpairs:
        pdf = plot_one_cellpair(pair, aggs[pair], out_dir)
        print(f"{pair}\trows_kept={rows_kept[pair]}\tpdf={pdf}")


if __name__ == "__main__":
    main()
