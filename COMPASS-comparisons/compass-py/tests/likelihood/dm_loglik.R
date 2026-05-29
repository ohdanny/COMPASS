if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}
library(jsonlite)

source("../../compass_core.R")

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- dirname(normalizePath(script_path))
json_path <- file.path(script_dir, "dm_loglik.json")

g <- fromJSON(json_path, simplifyMatrix = FALSE)

Y     <- do.call(rbind, lapply(g$inputs$Y, as.numeric))
Pi    <- do.call(rbind, lapply(g$inputs$Pi, as.numeric))
kappa <- g$inputs$kappa
N     <- rowSums(Y)

g$R_outputs$with_const    <- dm_loglik(Y, N, Pi, kappa, TRUE)
g$R_outputs$without_const <- dm_loglik(Y, N, Pi, kappa, FALSE)

write_json(g, json_path, pretty = TRUE, digits = 16, auto_unbox = TRUE)
cat("Wrote outputs to", json_path, "\n")
cat("  with_const    =", g$R_outputs$with_const, "\n")
cat("  without_const =", g$R_outputs$without_const, "\n")