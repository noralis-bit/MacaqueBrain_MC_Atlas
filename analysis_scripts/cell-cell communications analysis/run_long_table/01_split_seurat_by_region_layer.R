.libPaths(c("/sdd/bgi/wangzilu/miniconda3/envs/deeptalk-env/lib/R/library", .libPaths()))
suppressPackageStartupMessages(library(Seurat))

args <- commandArgs(trailingOnly = TRUE)
input_rds <- ifelse(length(args) >= 1, args[[1]], "adata286.seurat.rds")
out_dir <- ifelse(length(args) >= 2, args[[2]], "region_layer_rds")
manifest <- file.path(out_dir, "manifest.tsv")

safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  gsub("^_+|_+$", "", x)
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

message("Reading: ", input_rds)
obj <- readRDS(input_rds)
DefaultAssay(obj) <- "RNA"

needed <- c("region", "layer")
missing <- setdiff(needed, colnames(obj@meta.data))
if (length(missing) > 0) {
  stop("Missing metadata columns: ", paste(missing, collapse = ", "))
}

obj$region_layer <- paste(as.character(obj$region), as.character(obj$layer), sep = "-")
blocks <- sort(unique(as.character(obj$region_layer)))
message("Region-layer blocks: ", length(blocks))

rows <- list()
for (block in blocks) {
  idx <- which(as.character(obj$region_layer) == block)
  safe <- safe_name(block)
  out_file <- file.path(out_dir, paste0("adata286.", safe, ".seurat.rds"))
  message("Writing ", block, " cells=", length(idx), " -> ", out_file)
  sub <- obj[, idx]
  saveRDS(sub, out_file, compress = FALSE)
  rows[[length(rows) + 1]] <- data.frame(
    region_layer = block,
    safe_name = safe,
    n_cells = length(idx),
    file = out_file,
    stringsAsFactors = FALSE
  )
  rm(sub)
  gc(verbose = FALSE)
}

manifest_df <- do.call(rbind, rows)
write.table(manifest_df, manifest, sep = "\t", quote = FALSE, row.names = FALSE)
message("Manifest: ", manifest)
message("DONE")
