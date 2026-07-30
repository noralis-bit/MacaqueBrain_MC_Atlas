.libPaths(c("/sdd/bgi/wangzilu/miniconda3/envs/deeptalk-env/lib/R/library", .libPaths()))
suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
in_dir <- ifelse(
  length(args) >= 1,
  args[[1]],
  "/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/niches_region_layer_radius100_custom_human_raw_celltocellspatial"
)
out_file <- ifelse(
  length(args) >= 2,
  args[[2]],
  file.path(in_dir, "niches_region_layer_radius100_celltocellspatial_long_nonzero.csv.gz")
)
summary_file <- ifelse(
  length(args) >= 3,
  args[[3]],
  file.path(in_dir, "niches_region_layer_radius100_celltocellspatial_long_nonzero.summary.tsv")
)

files <- sort(list.files(in_dir, pattern = "^niches\\..*\\.rds$", full.names = TRUE))
if (length(files) == 0) {
  stop("No RDS files found in: ", in_dir)
}
if (file.exists(out_file)) file.remove(out_file)
if (file.exists(summary_file)) file.remove(summary_file)

meta_keep <- c(
  "SendingType", "ReceivingType", "VectorType",
  "region_name.Sending", "region.Sending", "layer.Sending", "annotation.Sending",
  "x.Sending", "y.Sending", "x_r.Sending", "y_r.Sending",
  "region_name.Receiving", "region.Receiving", "layer.Receiving", "annotation.Receiving",
  "x.Receiving", "y.Receiving", "x_r.Receiving", "y_r.Receiving",
  "region_layer.Sending", "region_layer.Receiving"
)

celltype_relabel <- c(
  "L3/5-IT.03" = "L4-IT.00",
  "L3/5-IT.04" = "L4-IT.01",
  "L3/5-IT.05" = "L4-IT.02"
)

relabel_celltype_columns <- function(dt) {
  cols <- intersect(
    c("SendingType", "ReceivingType", "annotation.Sending", "annotation.Receiving"),
    names(dt)
  )
  for (col in cols) {
    values <- as.character(dt[[col]])
    hit <- values %in% names(celltype_relabel)
    if (any(hit)) values[hit] <- unname(celltype_relabel[values[hit]])
    set(dt, j = col, value = values)
  }
  dt
}

split_event <- function(x) {
  parts <- tstrsplit(x, "—", fixed = TRUE, keep = 1:2)
  list(ligand = parts[[1]], receptor = parts[[2]])
}

split_edge <- function(x) {
  parts <- tstrsplit(x, "—", fixed = TRUE, keep = 1:2)
  list(sending_cell = parts[[1]], receiving_cell = parts[[2]])
}

header <- TRUE
total_rows <- 0L
summary_rows <- list()

message("Input files: ", length(files))
message("Output: ", out_file)

for (idx in seq_along(files)) {
  f <- files[[idx]]
  region_layer <- sub("^niches\\.", "", tools::file_path_sans_ext(basename(f)))
  message(sprintf("[%02d/%02d] reading %s", idx, length(files), basename(f)))

  obj <- readRDS(f)
  ctc <- obj$CellToCellSpatial
  if (is.null(ctc)) stop("CellToCellSpatial not found in: ", f)
  mat <- ctc$CellToCellSpatialMatrix
  meta <- as.data.table(ctc$metadata, keep.rownames = "edge_id")

  if (is.null(mat) || is.null(meta)) {
    stop("Missing matrix or metadata in: ", f)
  }

  nz <- as.data.table(summary(mat))
  setnames(nz, c("event_i", "edge_j", "score"))
  if (nrow(nz) == 0) {
    summary_rows[[length(summary_rows) + 1]] <- data.table(
      region_layer = region_layer,
      edges = ncol(mat),
      events = nrow(mat),
      nonzero_rows = 0L
    )
    next
  }

  event_names <- rownames(mat)[nz$event_i]
  edge_names <- colnames(mat)[nz$edge_j]
  ev <- split_event(event_names)
  edge <- split_edge(edge_names)

  dt <- data.table(
    region_layer_result = region_layer,
    edge_id = edge_names,
    sending_cell = edge$sending_cell,
    receiving_cell = edge$receiving_cell,
    event = event_names,
    ligand = ev$ligand,
    receptor = ev$receptor,
    score = nz$score
  )

  meta_sub <- meta[nz$edge_j, c("edge_id", intersect(meta_keep, names(meta))), with = FALSE]
  dt <- cbind(dt, meta_sub[, setdiff(names(meta_sub), "edge_id"), with = FALSE])
  dt <- relabel_celltype_columns(dt)

  fwrite(
    dt,
    file = out_file,
    append = !header,
    col.names = header,
    quote = "auto",
    nThread = max(1L, min(8L, parallel::detectCores(logical = TRUE)))
  )

  total_rows <- total_rows + nrow(dt)
  summary_rows[[length(summary_rows) + 1]] <- data.table(
    region_layer = region_layer,
    edges = ncol(mat),
    events = nrow(mat),
    nonzero_rows = nrow(dt)
  )
  message(sprintf("[%02d/%02d] edges=%s nonzero=%s cumulative=%s",
                  idx, length(files), ncol(mat), nrow(dt), total_rows))

  rm(obj, ctc, mat, meta, nz, dt, meta_sub, event_names, edge_names, ev, edge)
  gc(verbose = FALSE)
  header <- FALSE
}

summary_dt <- rbindlist(summary_rows)
summary_dt[, total_nonzero_rows := sum(nonzero_rows)]
fwrite(summary_dt, summary_file, sep = "\t")

message("DONE")
message("rows: ", total_rows)
message("summary: ", summary_file)
message("output: ", out_file)
