from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from adjustText import adjust_text


base_dir = Path(
    "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/"
    "niches_region_layer_radius100_custom_human_raw"
)
wide_csv = base_dir / "niches_region_name_radius100_custom_human_raw_wide.csv"
out_dir = base_dir / "adata_layer_lr_plots"
out_dir.mkdir(parents=True, exist_ok=True)

h5ad_out = out_dir / "adata_niche_radius100_lr_scores.h5ad"
dotplot_pdf = out_dir / "layer_LR_dotplot.pdf"
volcano_pdf = out_dir / "volcano_L3s_vs_L3d.pdf"
deg_csv = out_dir / "rank_genes_groups_l3s.csv"
summary_txt = out_dir / "adata_layer_lr_plots_summary.txt"

meta_cols = [
    "niche_id",
    "region_result",
    "orig.ident",
    "nCount_NeighborhoodToCell",
    "nFeature_NeighborhoodToCell",
    "ReceivingCell",
    "ReceivingType",
    "region_global_id",
    "region_name",
    "chip",
    "annotation",
    "batch",
    "region",
    "layer",
    "old_layer",
    "x",
    "y",
    "x_r",
    "y_r",
]

layer_order = ["l1", "l2", "l3s", "l3d", "l5", "l6"]
selected_lr = [
    "MAG-MAG",
    "PSAP-GPR37",
    "NRXN1-CLSTN2",
    "APP-SORL1",
    "TENM2-ADGRL1",
    "NRG1-ERBB4",
    "CADM3-CADM3",
    "CADM3-NECTIN3",
    "C1QL2-ADGRB3",
    "NRXN1-LRRTM4",
    "PSAP-GPR37L1",
    "PTN-PTPRZ1",
]


def normalize_event_name(x: str) -> str:
    return x.replace("—", "-")


if h5ad_out.exists():
    print(f"Reading existing AnnData: {h5ad_out}")
    adata = sc.read_h5ad(h5ad_out)
else:
    print(f"Building AnnData from: {wide_csv}")
    header = pd.read_csv(wide_csv, nrows=0)
    lr_cols_original = [c for c in header.columns if c not in meta_cols]
    lr_cols_norm = [normalize_event_name(c) for c in lr_cols_original]

    obs_parts = []
    x_parts = []
    total_rows = 0
    for chunk in pd.read_csv(wide_csv, chunksize=20000):
        obs = chunk.loc[:, meta_cols].copy()
        obs.index = obs["niche_id"].astype(str)
        x = sp.csr_matrix(chunk.loc[:, lr_cols_original].to_numpy(dtype=np.float32))
        obs_parts.append(obs)
        x_parts.append(x)
        total_rows += chunk.shape[0]
        print(f"processed_rows {total_rows}", flush=True)

    obs = pd.concat(obs_parts, axis=0)
    X = sp.vstack(x_parts, format="csr")
    var = pd.DataFrame(index=pd.Index(lr_cols_norm, name="LRpair"))
    var["original_event"] = lr_cols_original
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = adata.obs.loc[:, ["x", "y"]].to_numpy(dtype=np.float32)
    adata.write_h5ad(h5ad_out, compression="gzip")
    print(f"Saved AnnData: {h5ad_out}")

adata.obs["layer"] = adata.obs["layer"].astype(
    pd.CategoricalDtype(categories=layer_order, ordered=True)
)
adata = adata[adata.obs["layer"].notna()].copy()

print("Running rank_genes_groups...")
sc.tl.rank_genes_groups(
    adata,
    groupby="layer",
    method="wilcoxon",
    n_genes=min(5000, adata.n_vars),
    use_raw=False,
)
print("DEG done")

present_lr = [x for x in selected_lr if x in adata.var_names]
missing_lr = [x for x in selected_lr if x not in adata.var_names]

sc.pl.dotplot(
    adata,
    var_names=present_lr,
    groupby="layer",
    standard_scale="var",
    dot_max=0.25,
    cmap="Reds",
    figsize=(6, 2.5),
    show=False,
)
plt.savefig(dotplot_pdf, dpi=300, bbox_inches="tight")
plt.close("all")

result = adata.uns["rank_genes_groups"]
if "l3s" not in result["names"].dtype.names:
    raise RuntimeError("l3s not found in rank_genes_groups result")

pvals = result["pvals"]["l3s"]
min_pval = np.nextafter(0, 1)
pvals_clipped = np.where(pvals == 0, min_pval, pvals)
de_df = pd.DataFrame(
    {
        "gene": result["names"]["l3s"],
        "log2fc": result["logfoldchanges"]["l3s"],
        "pval": pvals,
        "pval_adj": result["pvals_adj"]["l3s"],
        "-log10(pval)": -np.log10(pvals_clipped),
    }
).dropna()
de_df.to_csv(deg_csv, index=False)

pval_threshold = 0.05
log2fc_threshold = 1.0
de_df["sig"] = "Not significant"
de_df.loc[
    (de_df["pval_adj"] < pval_threshold) & (de_df["log2fc"] > log2fc_threshold),
    "sig",
] = "L3s"
de_df.loc[
    (de_df["pval_adj"] < pval_threshold) & (de_df["log2fc"] < -log2fc_threshold),
    "sig",
] = "L3d"

colors = {
    "L3s": "#E74C3C",
    "L3d": "#3498DB",
    "Not significant": "#BDC3C7",
}

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.set_facecolor("white")
for category in ["Not significant", "L3d", "L3s"]:
    subset = de_df[de_df["sig"] == category]
    ax.scatter(
        subset["log2fc"],
        subset["-log10(pval)"],
        c=colors[category],
        s=40 if category == "Not significant" else 65,
        alpha=0.6 if category == "Not significant" else 0.9,
        edgecolor="white",
        linewidth=0.4,
        label=category,
        zorder=3 if category != "Not significant" else 2,
    )

ax.axhline(-np.log10(pval_threshold), color="#7F8C8D", linestyle="--", linewidth=1)
ax.axvline(log2fc_threshold, color="#7F8C8D", linestyle="--", linewidth=1)
ax.axvline(-log2fc_threshold, color="#7F8C8D", linestyle="--", linewidth=1)

texts = []
top_up = de_df[de_df["sig"] == "L3s"].nlargest(5, "-log10(pval)")
top_down = de_df[de_df["sig"] == "L3d"].nlargest(5, "-log10(pval)")
top_genes = pd.concat([top_up, top_down])
for _, row in top_genes.iterrows():
    texts.append(
        ax.text(
            row["log2fc"],
            row["-log10(pval)"],
            row["gene"],
            fontsize=9,
            color="#2C3E50",
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
    )
if texts:
    adjust_text(texts, arrowprops=dict(arrowstyle="->", lw=0.8, color="#7F8C8D"), ax=ax)

ax.set_xlabel("Log$_2$ Fold Change", fontsize=13, weight="bold")
ax.set_ylabel("-Log$_{10}$ P-value", fontsize=13, weight="bold")
ax.set_title("Volcano Plot: L3s vs L3d", fontsize=15, weight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(title="Significance", frameon=True, fancybox=True, framealpha=0.85, loc="upper right")
plt.tight_layout()
fig.savefig(volcano_pdf, dpi=300, bbox_inches="tight")
plt.close(fig)

summary = "\n".join(
    [
        f"h5ad {h5ad_out}",
        f"n_obs {adata.n_obs}",
        f"n_vars {adata.n_vars}",
        f"dotplot_pdf {dotplot_pdf}",
        f"volcano_pdf {volcano_pdf}",
        f"deg_csv {deg_csv}",
        "present_selected_LR " + ",".join(present_lr),
        "missing_selected_LR " + ",".join(missing_lr),
        "layers " + ",".join(map(str, adata.obs["layer"].cat.categories)),
    ]
)
summary_txt.write_text(summary + "\n")
print(summary)
