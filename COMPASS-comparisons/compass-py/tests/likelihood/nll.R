if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}
library(jsonlite)

source("../../compass_core.R")  # provides neg_loglik(...)

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- dirname(normalizePath(script_path))
json_path <- file.path(script_dir, "nll.json")

g <- fromJSON(json_path, simplifyMatrix = FALSE, simplifyVector = FALSE)

cases <- g$config$cases
seed  <- as.integer(g$config$seed)

results <- vector("list", length(cases))
for (i in seq_along(cases)) {
  cs   <- cases[[i]]
  K    <- as.integer(cs$K)
  Tnum <- as.integer(cs$T)
  r    <- as.integer(cs$r)
  full <- isTRUE(cs$full)
  S    <- as.integer(cs$S)

  Kminus <- K - 1L
  # mirrors compass.indexing.ParamIndex.n_param_per_k
  p_per <- if (full) Tnum * (1L + r) else Tnum + r
  total <- Kminus * p_per + 1L

  set.seed(seed + i)

  # Random inputs. Magnitudes kept small so eta = Z %*% Gamma stays inside
  # the [-40, 40] regime where R's neg_loglik does not clip (Python's nll
  # has no clipping; equality only holds in the un-clipped regime).
  Z      <- cbind(1, matrix(rnorm(S * (p_per - 1L), sd = 0.5), S, p_per - 1L))
  params <- rnorm(total, sd = 0.3)
  params[total] <- log(5)                       # log_kappa
  N      <- rpois(S, 6) + 1L
  Y      <- t(vapply(seq_len(S),
                     function(s) rmultinom(1, N[s], rep(1/K, K))[, 1],
                     integer(K)))

  val <- neg_loglik(params, Y, N, Z, K)         # <-- R function under test
  stopifnot(is.finite(val) && val < 1e11)       # guard didn't fire

  results[[i]] <- list(
    name = cs$name,
    K = K, T = Tnum, r = r, full = full, S = S,
    params = I(as.numeric(params)),
    Y = I(lapply(seq_len(nrow(Y)), function(s) as.numeric(Y[s, ]))),
    N = I(as.numeric(N)),
    Z = I(lapply(seq_len(nrow(Z)), function(s) as.numeric(Z[s, ]))),
    R_output = val
  )
}

g$R_outputs <- results

write_json(g, json_path, pretty = TRUE, digits = 17, auto_unbox = TRUE)
cat("Wrote outputs to", json_path, "\n")
for (rec in results) {
  cat(sprintf("  %-18s nll = %.10f\n", rec$name, rec$R_output))
}
