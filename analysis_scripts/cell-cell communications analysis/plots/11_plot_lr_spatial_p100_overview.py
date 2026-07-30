#!/usr/bin/env python3
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


WORK_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC"
)
LONG_TABLE = WORK_DIR / (
    "niches_region_layer_radius100_custom_human_raw_annotation_new_0609_"
    "celltocellspatial/niches_region_layer_radius100_annotation_new_0609_"
    "celltocellspatial_long_nonzero.csv.gz"
)
JSON_DIR = Path("/sdd/datasets/motor_spatial_cellbins/mq286_outline")
SAVE_DIR = WORK_DIR / "niches_lr_spatial_annotation_new_0609_radius100_P100"

EVENTS = [
    "APP-SORL1",
    "NRXN1-LRRTM4",
    "MAG-MAG",
    "CADM3-NECTIN3",
    "LRFN5-PTPRF",
    "CLDN11-CLDN11",
    "NRXN1-CLSTN2",
]
CHIPS = ["T575", "T572", "T582", "T580", "T578", "T574"]
CHIP_OFFSET = {
    "T575": 400000,
    "T572": 320000,
    "T582": 0,
    "T580": 80000,
    "T578": 160000,
    "T574": 240000,
}


def apply_correction_with_matrix(points, corr_para):
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(corr_para["transformation_matrix"]).reshape(3, 3)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ matrix.T)[:, :2]


def load_contours(chip):
    with open(JSON_DIR / f"MQ286-{chip}-P0.json") as handle:
        data = json.load(handle)

    contours = []
    for line in data.get("lines", []):
        points = (
            line.get("countours")
            or line.get("contours")
            or line.get("points")
            or []
        )
        if not points:
            continue
        points = np.asarray(points, dtype=float)
        if data.get("corr_para") is not None:
            points = apply_correction_with_matrix(points, data["corr_para"])
        contours.append(points)
    return contours


def read_long_table():
    usecols = [
        "sending_cell",
        "ligand",
        "receptor",
        "score",
        "x.Sending",
        "y.Sending",
    ]
    wanted = set(EVENTS)
    positions = []
    selected = []

    for chunk in pd.read_csv(
        LONG_TABLE,
        usecols=usecols,
        chunksize=500_000,
        low_memory=False,
    ):
        chunk["chip"] = chunk["sending_cell"].str.extract(r"^(T\d+)-", expand=False)
        chunk = chunk[chunk["chip"].isin(CHIPS)]
        positions.append(
            chunk[["sending_cell", "chip", "x.Sending", "y.Sending"]]
            .drop_duplicates("sending_cell")
        )

        event = chunk["ligand"].astype(str) + "-" + chunk["receptor"].astype(str)
        mask = event.isin(wanted)
        if mask.any():
            target = chunk.loc[mask].copy()
            target["interaction"] = event.loc[mask]
            selected.append(target)

    positions = pd.concat(positions, ignore_index=True).drop_duplicates("sending_cell")
    selected = pd.concat(selected, ignore_index=True)
    strengths = (
        selected.groupby(["interaction", "chip", "sending_cell"], as_index=False)
        .agg(
            plot_value=("score", "sum"),
            x=("x.Sending", "first"),
            y=("y.Sending", "first"),
            n_edges=("score", "size"),
        )
    )
    positions = positions.rename(columns={"x.Sending": "x", "y.Sending": "y"})
    return positions, strengths


def draw_panel(ax, background, values, contours, offset, vmax, title, colorbar=True):
    bg_x = background["x"].to_numpy() - offset
    bg_y = background["y"].to_numpy()
    ax.scatter(bg_x, -bg_y, s=2, c="#e6e6e6", edgecolors="none", zorder=1)

    if not values.empty:
        values = values.sort_values("plot_value")
        x = values["x"].to_numpy() - offset
        y = values["y"].to_numpy()
        clipped = np.minimum(values["plot_value"].to_numpy(), vmax)
        scatter = ax.scatter(
            x,
            -y,
            c=clipped,
            cmap="magma_r",
            vmin=0,
            vmax=vmax,
            s=5,
            edgecolors="none",
            alpha=1,
            zorder=2,
        )
        if colorbar:
            cax = inset_axes(ax, width="4%", height="30%", loc="center right", borderpad=1)
            plt.colorbar(scatter, cax=cax)

    all_contour_points = []
    for points in contours:
        ax.plot(points[:, 0], -points[:, 1], color="black", lw=0.5, alpha=0.9, zorder=3)
        all_contour_points.append(points)

    if all_contour_points:
        points = np.vstack(all_contour_points)
        margin = 1000
        ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
        ax.set_ylim(-points[:, 1].max() - margin, -points[:, 1].min() + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main_spatial_lr():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    positions, strengths = read_long_table()
    contours = {chip: load_contours(chip) for chip in CHIPS}

    summary = (
        strengths.groupby(["interaction", "chip"], as_index=False)
        .agg(n_sending_cells=("sending_cell", "nunique"), n_edges=("n_edges", "sum"), total_score=("plot_value", "sum"))
    )
    summary.to_csv(SAVE_DIR / "selected_lr_plot_summary.csv", index=False)

    overview_path = SAVE_DIR / "six_lr_six_chip_spatial_overview.pdf"
    with PdfPages(overview_path) as pdf:
        fig, axes = plt.subplots(len(EVENTS), len(CHIPS), figsize=(18, 22))
        for row, event in enumerate(EVENTS):
            event_values = strengths[strengths["interaction"] == event]
            # P100: retain the full score range without clipping.
            vmax = float(event_values["plot_value"].max())
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            for col, chip in enumerate(CHIPS):
                chip_bg = positions[positions["chip"] == chip]
                chip_values = event_values[event_values["chip"] == chip]
                draw_panel(
                    axes[row, col],
                    chip_bg,
                    chip_values,
                    contours[chip],
                    CHIP_OFFSET[chip],
                    vmax,
                    f"{event}\n{chip}",
                    colorbar=False,
                )

                fig_one, ax_one = plt.subplots(figsize=(8, 8))
                draw_panel(
                    ax_one,
                    chip_bg,
                    chip_values,
                    contours[chip],
                    CHIP_OFFSET[chip],
                    vmax,
                    f"{event} ({chip})",
                    colorbar=False,
                )
                fig_one.tight_layout()
                filename = f"all_legend_interaction_{event}_{chip}_final.png"
                fig_one.savefig(SAVE_DIR / filename, dpi=300, bbox_inches="tight")
                plt.close(fig_one)

        fig.suptitle("NICHES radius 100: spatial communication strength (P100)", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Long table: {LONG_TABLE}")
    print(summary.to_string(index=False))
    print(f"Saved 36 PNG files and overview PDF to: {SAVE_DIR}")




# ===== APP-SORL1 P100 overview; appended from plot_niches_app_sorl1_p100.py =====
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

EVENT = "APP-SORL1"


def main_app_sorl1():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    positions, strengths = read_long_table()
    values = strengths[strengths["interaction"] == EVENT].copy()
    if values.empty:
        raise RuntimeError(f"No values found for {EVENT}")

    vmax = float(values["plot_value"].max())
    contours = {chip: load_contours(chip) for chip in CHIPS}

    fig, axes = plt.subplots(1, len(CHIPS), figsize=(18, 4.5))
    for ax, chip in zip(axes, CHIPS):
        chip_bg = positions[positions["chip"] == chip]
        chip_values = values[values["chip"] == chip]
        draw_panel(
            ax,
            chip_bg,
            chip_values,
            contours[chip],
            CHIP_OFFSET[chip],
            vmax,
            f"{EVENT}\n{chip}",
            colorbar=False,
        )

        fig_one, ax_one = plt.subplots(figsize=(8, 8))
        draw_panel(
            ax_one,
            chip_bg,
            chip_values,
            contours[chip],
            CHIP_OFFSET[chip],
            vmax,
            f"{EVENT} ({chip})",
            colorbar=False,
        )
        fig_one.tight_layout()
        fig_one.savefig(
            SAVE_DIR / f"all_legend_interaction_{EVENT}_{chip}_final.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_one)

    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, vmax), cmap="magma_r"),
        ax=axes.tolist(),
        fraction=0.012,
        pad=0.015,
        aspect=25,
    )
    colorbar.set_label("Summed score (P100)")
    fig.suptitle(f"{EVENT}: spatial communication strength (P100)", fontsize=15)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.03, top=0.88, wspace=0.05)
    fig.savefig(SAVE_DIR / f"{EVENT}_six_chip_spatial_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    app_summary = (
        values.groupby(["interaction", "chip"], as_index=False)
        .agg(
            n_sending_cells=("sending_cell", "nunique"),
            n_edges=("n_edges", "sum"),
            total_score=("plot_value", "sum"),
        )
    )
    summary_path = SAVE_DIR / "selected_lr_plot_summary.csv"
    if summary_path.exists():
        old_summary = pd.read_csv(summary_path)
        old_summary = old_summary[old_summary["interaction"] != EVENT]
        app_summary = pd.concat([old_summary, app_summary], ignore_index=True)
    app_summary.sort_values(["interaction", "chip"]).to_csv(summary_path, index=False)

    print(app_summary[app_summary["interaction"] == EVENT].to_string(index=False))
    print(f"P100 vmax: {vmax:.10g}")
    print(f"Saved 6 PNG files and APP-SORL1 overview PDF to: {SAVE_DIR}")




def main():
    main_spatial_lr()
    main_app_sorl1()


if __name__ == "__main__":
    main()
