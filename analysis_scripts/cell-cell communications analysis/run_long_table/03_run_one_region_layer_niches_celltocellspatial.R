.libPaths(c("/sdd/bgi/wangzilu/miniconda3/envs/deeptalk-env/lib/R/library", .libPaths()))
suppressPackageStartupMessages({
  library(Seurat)
  library(NICHES)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("Usage: Rscript 03_run_one_region_layer_niches_celltocellspatial.R input_region_layer.rds out_prefix radius database species output_format custom_lr.csv")
}

input_rds <- args[[1]]
out_prefix <- args[[2]]
radius <- as.numeric(args[[3]])
database <- args[[4]]
species <- args[[5]]
output_format <- args[[6]]
custom_db_path <- args[[7]]

custom_lr <- NULL
if (database == "custom") {
  custom_lr <- read.csv(custom_db_path, check.names = FALSE)
  custom_lr <- custom_lr[, 1:2, drop = FALSE]
  custom_lr <- unique(custom_lr[complete.cases(custom_lr), , drop = FALSE])
}

message("Reading region-layer RDS: ", input_rds)
obj <- readRDS(input_rds)
DefaultAssay(obj) <- "RNA"
Idents(obj) <- obj$annotation

if (!"region_layer" %in% colnames(obj@meta.data)) {
  obj$region_layer <- paste(as.character(obj$region), as.character(obj$layer), sep = "-")
}
region_layer <- unique(as.character(obj$region_layer))
if (length(region_layer) != 1) region_layer <- paste(region_layer, collapse = ",")

message("Running NICHES CellToCellSpatial region_layer=", region_layer, " cells=", ncol(obj))
message("database=", database, " species=", species, " radius=", radius, " output_format=", output_format)
set.seed(2026)

niches <- RunNICHES(
  object = obj,
  assay = "RNA",
  LR.database = database,
  species = species,
  custom_LR_database = custom_lr,
  meta.data.to.map = c(
    "region_global_id", "region_name", "chip", "annotation", "batch",
    "region", "layer", "old_layer", "x", "y", "x_r", "y_r", "region_layer"
  ),
  position.x = "x",
  position.y = "y",
  cell_types = "annotation",
  k = NULL,
  rad.set = radius,
  CellToCell = FALSE,
  CellToSystem = FALSE,
  SystemToCell = FALSE,
  CellToCellSpatial = TRUE,
  CellToNeighborhood = FALSE,
  NeighborhoodToCell = FALSE,
  output_format = output_format
)

out_rds <- paste0(out_prefix, ".rds")
saveRDS(niches, out_rds, compress = FALSE)

summary_file <- paste0(out_prefix, ".summary.txt")
sink(summary_file)
cat("input_rds", input_rds, "\n")
cat("region_layer", region_layer, "\n")
cat("cells", ncol(obj), "\n")
cat("database", database, "\n")
cat("species", species, "\n")
cat("radius", radius, "\n")
cat("output_format", output_format, "\n")
if (!is.null(custom_lr)) cat("custom_lr_pairs", nrow(custom_lr), "\n")
cat("output_class", paste(class(niches), collapse = ","), "\n")
cat("names", paste(names(niches), collapse = ","), "\n")
for (nm in names(niches)) {
  x <- niches[[nm]]
  cat("object", nm, "\n")
  cat("  class", paste(class(x), collapse = ","), "\n")
  if (!is.null(dim(x))) cat("  dim", paste(dim(x), collapse = "x"), "\n")
  if (is.list(x)) {
    cat("  list_names", paste(names(x), collapse = ","), "\n")
    if (!is.null(x$CellToCellSpatialMatrix)) {
      cat("  matrix_class", paste(class(x$CellToCellSpatialMatrix), collapse = ","), "\n")
      cat("  matrix_dim", paste(dim(x$CellToCellSpatialMatrix), collapse = "x"), "\n")
    }
    if (!is.null(x$metadata)) {
      cat("  metadata_class", paste(class(x$metadata), collapse = ","), "\n")
      cat("  metadata_dim", paste(dim(x$metadata), collapse = "x"), "\n")
      cat("  metadata_cols", paste(colnames(x$metadata), collapse = ","), "\n")
    }
  }
  cat("  size_gb", round(as.numeric(object.size(x)) / 1024^3, 3), "\n")
}
sink()

message("Saved: ", out_rds)
message("Summary: ", summary_file)
message("DONE")
