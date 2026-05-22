source("../../compass_core.R")
library(jsonlite)

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("^--file=", "",
                       args[grep("^--file=", args)])))
json_path <- file.path(script_dir, "param_index.json")

g <- if (file.exists(json_path) && file.info(json_path)$size > 0) {
  tryCatch(
    fromJSON(json_path, simplifyVector = FALSE),
    error = function(e) {
      warning("param_index.json was unparseable; starting fresh: ", conditionMessage(e))
      list()
    }
  )
} else {
  list()
}

# Make sure g is a list even if the JSON was a scalar/array
if (!is.list(g)) g <- list()

write_case <- function(K, T, r, full) {
  idx <- param_index(K, T, r, full = full)
  list(
    K = K, T = T, r = r, full = full,
    p_per = idx$p_per,
    total = idx$total,
    alpha = I(idx$alpha),
    beta = I(lapply(idx$beta, I)),
    xi0 = I(lapply(idx$xi0, I)),
    xi_int = if (!is.null(idx$xi_int)) I(lapply(idx$xi_int, I)) else NULL,
    kappa = idx$kappa
  )
}

g$R_outputs <- list(
  case_reduced = write_case(3, 4, 5, FALSE),
  case_full    = write_case(3, 4, 5, TRUE),
  case_T2      = write_case(4, 2, 3, TRUE),
  case_K2      = write_case(2, 3, 4, TRUE),
  case_T1      = write_case(3, 1, 5, TRUE),   
  case_T1_red  = write_case(3, 1, 5, FALSE)  
)

write_json(g, json_path, pretty = TRUE, auto_unbox = TRUE)
cat("wrote", json_path, "\n")