if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}
library(jsonlite)

source("../../compass_core.R")

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- dirname(normalizePath(script_path))
json_path <- file.path(script_dir, "softmax_ref.json")

g <- fromJSON(json_path, simplifyMatrix = FALSE, simplifyVector = FALSE)

results <- vector("list", length(g$cases))
for (i in seq_along(g$cases)) {
  cs <- g$cases[[i]]
  eta <- do.call(rbind, lapply(cs$eta, as.numeric))
  Pi <- softmax_ref(eta)
  results[[i]] <- list(
    name = cs$name,
    Pi = I(lapply(seq_len(nrow(Pi)), function(s) as.numeric(Pi[s, ])))
  )
}

g$R_outputs <- results

write_json(g, json_path, pretty = TRUE, digits = 17, auto_unbox = TRUE)
cat("Wrote outputs to", json_path, "\n")
for (rec in results) cat("  case", rec$name, ": Pi computed\n")
