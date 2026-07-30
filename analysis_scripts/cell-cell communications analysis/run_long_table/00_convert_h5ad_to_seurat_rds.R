.libPaths(c("/sdd/bgi/wangzilu/miniconda3/envs/deeptalk-env/lib/R/library", .libPaths()))
suppressPackageStartupMessages({
  library(hdf5r)
  library(Matrix)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
infile <- ifelse(
  length(args) >= 1,
  args[[1]],
  "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/44.cross_main_region_cellchat/adata286.h5ad"
)
outfile <- ifelse(
  length(args) >= 2,
  args[[2]],
  "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/adata286.seurat.rds"
)
n_limit <- ifelse(length(args) >= 3, as.integer(args[[3]]), NA_integer_)

safe_exists <- function(h5, path) {
  isTRUE(tryCatch(h5$exists(path), error = function(e) FALSE))
}

read_h5ad_vector <- function(h5, path, n = NULL) {
  if (!safe_exists(h5, path)) return(NULL)

  if (safe_exists(h5, paste0(path, "/codes")) &&
      safe_exists(h5, paste0(path, "/categories"))) {
    codes <- as.integer(h5[[paste0(path, "/codes")]][])
    cats <- as.character(h5[[paste0(path, "/categories")]][])
    vals <- rep(NA_character_, length(codes))
    ok <- !is.na(codes) & codes >= 0L
    vals[ok] <- cats[codes[ok] + 1L]
  } else {
    vals <- as.vector(h5[[path]][])
  }

  if (!is.null(n)) vals <- vals[seq_len(n)]
  vals
}

h5 <- H5File$new(infile, mode = "r")
on.exit(h5$close_all())

all_cells <- read_h5ad_vector(h5, "obs/_index")
all_genes <- read_h5ad_vector(h5, "var/_index")
n_obs <- as.integer(length(all_cells))
n_var <- as.integer(length(all_genes))
if (!is.na(n_limit)) n_obs <- min(n_obs, n_limit)

message("Reading CSR matrix: cells=", n_obs, " genes=", n_var)
p <- h5[["X/indptr"]][1:(n_obs + 1)]
nnz <- as.integer(p[length(p)])
message("Nonzero values in selected cells: ", nnz)

i <- h5[["X/indices"]][1:nnz]
x <- h5[["X/data"]][1:nnz]
mat <- sparseMatrix(
  i = as.integer(i),
  p = as.integer(p),
  x = as.numeric(x),
  dims = c(n_var, n_obs),
  index1 = FALSE
)

rownames(mat) <- make.unique(as.character(all_genes))
cell_names <- as.character(all_cells[seq_len(n_obs)])
colnames(mat) <- cell_names

message("Building metadata")
md <- data.frame(row.names = cell_names)
obs_cols <- c(
  "region_global_id", "region_name", "chip", "annotation", "batch",
  "region", "layer", "old_layer"
)
for (nm in obs_cols) {
  vals <- read_h5ad_vector(h5, paste0("obs/", nm), n_obs)
  if (!is.null(vals)) md[[nm]] <- vals
}

if (safe_exists(h5, "obsm/spatial")) {
  sp <- h5[["obsm/spatial"]]$read()
  if (nrow(sp) != n_obs && ncol(sp) >= n_obs) sp <- t(sp)
  sp <- sp[seq_len(n_obs), , drop = FALSE]
  md$x <- sp[, 1]
  md$y <- sp[, 2]
}
if (safe_exists(h5, "obsm/spatial_r")) {
  spr <- h5[["obsm/spatial_r"]]$read()
  if (nrow(spr) != n_obs && ncol(spr) >= n_obs) spr <- t(spr)
  spr <- spr[seq_len(n_obs), , drop = FALSE]
  md$x_r <- spr[, 1]
  md$y_r <- spr[, 2]
}

message("Creating Seurat object with RNA counts and data layers")
obj <- CreateSeuratObject(counts = mat, assay = "RNA", meta.data = md, project = "adata286")
obj <- SetAssayData(obj, assay = "RNA", layer = "data", new.data = mat)
DefaultAssay(obj) <- "RNA"
if ("annotation" %in% colnames(obj@meta.data)) Idents(obj) <- obj$annotation

obj@misc$source_h5ad <- infile
obj@misc$conversion_note <- paste(
  "Converted from h5ad CSR X into Seurat RNA counts and data layers.",
  "Source X and layers/data were identical."
)

message("Object cells=", ncol(obj), " features=", nrow(obj), " layers=", paste(Layers(obj[["RNA"]]), collapse = ","))
message("Saving: ", outfile)
saveRDS(obj, outfile, compress = FALSE)
message("DONE")
