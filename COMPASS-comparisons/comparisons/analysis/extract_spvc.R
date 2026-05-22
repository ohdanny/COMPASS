# Run with the right LD_LIBRARY_PATH so libR can find flexiblas/icu, e.g.
#   FLEXI=/opt/software-current/2023.06/x86_64/amd/zen4/software/FlexiBLAS/3.3.1-GCC-12.3.0/lib
#   ICU=/opt/software-current/2023.06/x86_64/amd/zen4/software/ICU/73.2-GCCcore-12.3.0/lib
#   LD_LIBRARY_PATH="$FLEXI:$ICU:$LD_LIBRARY_PATH" Rscript extract_spvc.R

.libPaths(c("/mnt/scratch/hegazyab/guanao/svAPA/R/library", .libPaths()))

args  <- commandArgs(trailingOnly = FALSE)
fkey  <- "--file="
fpath <- sub(fkey, "", args[grep(fkey, args)])
here  <- if (length(fpath)) normalizePath(dirname(fpath)) else getwd()
RES_DIR  <- normalizePath(file.path(here, "..", "res"))
OUT_PATH <- file.path(here, "spvc_pvals.csv")

scenarios <- list(
  list(id = 0L, expr = "nonspatial", iso = "strict_null"),
  list(id = 1L, expr = "nonspatial", iso = "shared_null"),
  list(id = 2L, expr = "nonspatial", iso = "single_interaction"),
  list(id = 3L, expr = "nonspatial", iso = "two_interaction"),
  list(id = 4L, expr = "spatial",    iso = "strict_null"),
  list(id = 5L, expr = "spatial",    iso = "shared_null"),
  list(id = 6L, expr = "spatial",    iso = "single_interaction"),
  list(id = 7L, expr = "spatial",    iso = "two_interaction")
)

collect <- function(pv, sid, expr, iso, rep_id, k, model) {
  if (is.null(pv) || length(pv) == 0) return(NULL)
  data.frame(
    scenario_id = sid, expr = expr, iso = iso, rep = rep_id,
    k = k, model = model, term = names(pv),
    pval = as.numeric(pv),
    stringsAsFactors = FALSE
  )
}

rows <- list()
for (s in scenarios) {
  path <- file.path(RES_DIR, s$expr, s$iso, "spvc_results.rds")
  if (!file.exists(path)) { message("missing: ", path); next }
  x <- readRDS(path)
  for (rep_id in seq_along(x)) {
    r <- x[[rep_id]]
    for (iso_name in names(r$results.constant)) {
      k <- as.integer(sub("iso", "", iso_name)) - 1L
      rows[[length(rows) + 1L]] <- collect(
        r$results.constant[[iso_name]]$p.value,
        s$id, s$expr, s$iso, rep_id, k, "constant"
      )
    }
    for (iso_name in names(r$results.varying)) {
      k <- as.integer(sub("iso", "", iso_name)) - 1L
      rows[[length(rows) + 1L]] <- collect(
        r$results.varying[[iso_name]]$p.value,
        s$id, s$expr, s$iso, rep_id, k, "varying"
      )
    }
  }
  cat(sprintf("read scenario %d %s/%s (%d reps)\n", s$id, s$expr, s$iso, length(x)))
}

out <- do.call(rbind, rows)
write.csv(out, OUT_PATH, row.names = FALSE)
cat("wrote", nrow(out), "rows to", OUT_PATH, "\n")
