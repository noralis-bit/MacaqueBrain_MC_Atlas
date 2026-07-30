#!/usr/bin/env python3
import json
from pathlib import Path

import anndata as ad
import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from range_compression import RangeCompressedMask
from scipy.spatial.distance import cdist


H5AD = "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/44.cross_main_region_cellchat/adata286.h5ad"
OUT_DIR = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw_celltocellspatial/colocalization_cellmask"
)
RCM_PATH = "/sdd/datasets/motor_cellmask/T578-cell-range-compression/"
CHIP = "T578"

CELLTYPE_RELABEL = {
    "L3/5-IT.03": "L4-IT.00",
    "L3/5-IT.04": "L4-IT.01",
    "L3/5-IT.05": "L4-IT.02",
}

PAIR_CONFIGS = {
    "Ast.00_L2/3-IT.04": {
        "celltypes": ["Ast.00", "L2/3-IT.04"],
        "zoom_region": (200400, 201400, 15000, 16000),
    },
    "Ast.00_L4-IT.02": {
        "celltypes": ["Ast.00", "L4-IT.02"],
        "zoom_region": (200300, 201300, 13000, 14000),
    },
}


def safe_name(text):
    return text.replace("/", "_").replace("|", "_").replace(" ", "")


def plot_colocalization_celltypes(
    adata,
    celltypes,
    celltype_key="annotation",
    point_size=0.5,
    figsize=(5, 5),
    celltype_colors=("#FF6B6B", "#4ECDC4"),
    background_color="#DCDCDC",
    distance_threshold=100,
    zoom_region=None,
    show_legend=True,
    invert_y=False,
):
    coords = adata.obsm["spatial"]
    ct = adata.obs[celltype_key].astype(str)

    if zoom_region is not None:
        x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom = zoom_region
        region_mask = (
            (coords[:, 0] >= x_min_zoom)
            & (coords[:, 0] <= x_max_zoom)
            & (coords[:, 1] >= y_min_zoom)
            & (coords[:, 1] <= y_max_zoom)
        )
        adata = adata[region_mask].copy()
        coords = adata.obsm["spatial"]
        ct = adata.obs[celltype_key].astype(str)

    idx1 = np.asarray(ct == celltypes[0])
    idx2 = np.asarray(ct == celltypes[1])
    coords1 = coords[idx1]
    coords2 = coords[idx2]

    if len(coords1) > 0 and len(coords2) > 0:
        dist_matrix = cdist(coords1, coords2)
        close_pairs = np.where(dist_matrix < distance_threshold)
        idx1_keep = np.unique(np.where(idx1)[0][close_pairs[0]])
        idx2_keep = np.unique(np.where(idx2)[0][close_pairs[1]])
    else:
        idx1_keep = np.array([], dtype=int)
        idx2_keep = np.array([], dtype=int)

    keep_mask = np.zeros(len(adata), dtype=bool)
    keep_mask[idx1_keep] = True
    keep_mask[idx2_keep] = True

    zoom_ids = {
        "all": adata.obs.index.astype(str).tolist(),
        "celltype1_coloc": adata.obs.index[idx1_keep].astype(str).tolist(),
        "celltype2_coloc": adata.obs.index[idx2_keep].astype(str).tolist(),
    }

    coords_plot = coords.copy()
    if invert_y:
        coords_plot[:, 1] = -coords_plot[:, 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    other_mask = ~(idx1 | idx2)
    ax.scatter(
        coords_plot[other_mask, 0],
        coords_plot[other_mask, 1],
        c=background_color,
        s=point_size * 10,
        alpha=0.6,
        edgecolors="none",
    )

    only_one_mask = (idx1 | idx2) & ~keep_mask
    ax.scatter(
        coords_plot[only_one_mask, 0],
        coords_plot[only_one_mask, 1],
        c=background_color,
        s=point_size * 10,
        alpha=0.8,
        edgecolors="none",
    )

    ax.scatter(
        coords_plot[idx1_keep, 0],
        coords_plot[idx1_keep, 1],
        c=celltype_colors[0],
        s=point_size * 10,
        edgecolors="none",
    )
    ax.scatter(
        coords_plot[idx2_keep, 0],
        coords_plot[idx2_keep, 1],
        c=celltype_colors[1],
        s=point_size * 10,
        edgecolors="none",
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{celltypes[0]} -> {celltypes[1]}", fontsize=10)

    if show_legend:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=celltype_colors[0], edgecolor="none", label=celltypes[0]),
            Patch(facecolor=celltype_colors[1], edgecolor="none", label=celltypes[1]),
        ]
        ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

    fig.tight_layout()
    return fig, zoom_ids


@nb.njit(parallel=True)
def filter_mask(mask, keep_ids):
    out = np.zeros_like(mask)
    keep_set = set(keep_ids)
    for i in nb.prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] in keep_set:
                out[i, j] = mask[i, j]
    return out


def save_cellmask_png(zoom_ids, pair_key):
    rcm1 = RangeCompressedMask.load(RCM_PATH)
    t1 = rcm1.to_mask()

    ids1 = set(zoom_ids[pair_key]["celltype1_coloc"])
    ids2 = set(zoom_ids[pair_key]["celltype2_coloc"])
    ids_all = set(zoom_ids[pair_key]["all"])
    ids1 = set(int(x) for x in ids1)
    ids2 = set(int(x) for x in ids2)
    ids_all = set(int(x) for x in ids_all)
    ids_keep = ids1 | ids2 | ids_all

    t1_filtered = filter_mask(t1, np.array(list(ids_keep), dtype=t1.dtype))

    ys, xs = np.where(t1_filtered > 0)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    t1_crop = t1_filtered[ymin : ymax + 1, xmin : xmax + 1]
    mask_rgb = np.ones((*t1_crop.shape, 3), dtype=np.uint8) * 255

    mask_rgb[np.isin(t1_crop, list(ids_all))] = [234, 234, 234]
    mask_rgb[np.isin(t1_crop, list(ids1))] = [255, 107, 107]
    mask_rgb[np.isin(t1_crop, list(ids2))] = [79, 205, 196]

    out_path = OUT_DIR / f"T578_coloc_onlypoints_{safe_name(pair_key)}.png"
    success = cv2.imwrite(str(out_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    print(f"mask_rgb shape: {mask_rgb.shape}")
    print(f"mask_rgb dtype: {mask_rgb.dtype}")
    print(f"mask_rgb min/max values: {mask_rgb.min()}/{mask_rgb.max()}")

    if not success:
        print(f"cv2.imwrite failed for {out_path}")
    else:
        print(f"Image saved successfully to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata286 = ad.read_h5ad(H5AD)
    adata286.obs["annotation"] = adata286.obs["annotation"].astype(str).replace(CELLTYPE_RELABEL)
    adata_t578 = adata286[adata286.obs["chip"].astype(str) == CHIP].copy()

    summary_rows = []
    for pair_key, cfg in PAIR_CONFIGS.items():
        fig, pair_zoom_ids = plot_colocalization_celltypes(
            adata_t578,
            celltype_key="annotation",
            celltypes=cfg["celltypes"],
            point_size=0.5,
            figsize=(5, 5),
            celltype_colors=["#FF6B6B", "#4ECDC4"],
            background_color="#DCDCDC",
            distance_threshold=100,
            zoom_region=cfg["zoom_region"],
            show_legend=True,
        )

        fig.savefig(OUT_DIR / f"{safe_name(pair_key)}_colocalization_points.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{safe_name(pair_key)}_colocalization_points.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        zoom_ids = {pair_key: pair_zoom_ids}
        for key, subdict in zoom_ids.items():
            for subkey, id_list in subdict.items():
                subdict[subkey] = [s.replace("T578-cell-", "") for s in id_list]

        (OUT_DIR / f"{safe_name(pair_key)}_zoom_ids.json").write_text(json.dumps(zoom_ids, indent=2) + "\n")
        save_cellmask_png(zoom_ids, pair_key)

        summary_rows.append(
            {
                "pair": pair_key,
                "celltype1": cfg["celltypes"][0],
                "celltype2": cfg["celltypes"][1],
                "zoom_region": cfg["zoom_region"],
                "n_all": len(zoom_ids[pair_key]["all"]),
                "n_celltype1_coloc": len(zoom_ids[pair_key]["celltype1_coloc"]),
                "n_celltype2_coloc": len(zoom_ids[pair_key]["celltype2_coloc"]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "09_colocalization_cellmask_summary.csv", index=False)


if __name__ == "__main__":
    main()
