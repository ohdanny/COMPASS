# =============================================================================
# COMPASS demo simulation: three toy scenarios + five sanity checks
#
# Scenarios (single script, single gene):
#   1. null                 : Xi = 0.  Only baseline / spatially constant
#                             cell-type-composition effect / shared trend.
#   2. single_interaction   : one cell-type-contrast has a spatially varying
#                             effect on one category (xi_{1, 1}).
#   3. two_interaction      : two contrasts active with different spatial
#                             patterns (xi_{1,1}, xi_{2,2}).
#
# Sanity checks (debugging level):
#   (1) optimizer converges
#   (2) fitted probabilities stay in (0, 1)
#   (3) omnibus p-value small under null, large under alternative
#       (i.e. small = significant under alt, "small under null" = uniform)
#   (4) Wald test stronger for the active block than the inactive blocks
#   (5) changing basis dimension r does not destabilize the fit
#
# Outputs go to results/20260420_demo_simulation/.
# =============================================================================


suppressPackageStartupMessages({
  library(splines)
})


# Resolve script directory so we can source compass_core.R reliably.
resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  frames <- sys.frames()
  for (f in rev(frames)) {
    of <- f$ofile
    if (!is.null(of) && nzchar(of)) return(dirname(normalizePath(of)))
  }
  "./"
}
SCRIPT_DIR <- resolve_script_dir()
source(file.path(SCRIPT_DIR, "compass_core.R"))


# ---------- Output directory & logger ----------------------------------------


OUT_DIR <- "./Rres"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
LOG_FILE <- file.path(OUT_DIR, "run.log")
if (file.exists(LOG_FILE)) file.remove(LOG_FILE)


log_msg <- function(...) {
  msg <- paste0(format(Sys.time(), "%H:%M:%S"), " | ",
                paste0(..., collapse = ""))
  cat(msg, "\n", sep = "", file = LOG_FILE, append = TRUE)
  cat(msg, "\n", sep = "")
}


log_msg("=== COMPASS demo simulation ===")
log_msg("R version: ", R.version.string)
log_msg("Output dir: ", OUT_DIR)


# ---------- Shared setup -----------------------------------------------------


set.seed(20260420)


GRID_N <- 20L
S <- GRID_N * GRID_N
coords <- as.matrix(expand.grid(u = seq(0, 1, length.out = GRID_N),
                                v = seq(0, 1, length.out = GRID_N)))


# Cell-type composition W: T = 3 cell types. Two Gaussian bumps set the
# spatial _mean_, and each spot's composition is drawn from a Dirichlet
# around that mean so that X varies at a finer spatial scale than B. This
# keeps corr(B, X*B) well below 1 and makes the efficient information
# non-singular (see comment in _diagnose.R).
gauss_bump <- function(coords, ctr, sig) {
  d2 <- (coords[, 1] - ctr[1])^2 + (coords[, 2] - ctr[2])^2
  exp(-d2 / (2 * sig^2))
}
W_smooth_raw <- cbind(0.8 + 2.5 * gauss_bump(coords, c(0.25, 0.25), 0.22),
                      0.8 + 2.5 * gauss_bump(coords, c(0.75, 0.75), 0.22),
                      1.0)
W_smooth <- W_smooth_raw / rowSums(W_smooth_raw)
W_concentration <- 30
W <- t(apply(W_smooth, 1, function(p) {
  x <- rgamma(length(p), shape = W_concentration * p + 0.01, rate = 1)
  x / sum(x)
}))
Tnum <- ncol(W)


K <- 3L                          # transcript-processing categories
Kminus <- K - 1L


# Spatial basis and design (default r = 4 via tensor-product cubic splines)
B_default <- build_bspline_basis(coords, df_per_axis = 2L,
                                 orthonormalize = TRUE)
r_default <- ncol(B_default)
X_default <- W[, -Tnum, drop = FALSE]


log_msg(sprintf("S=%d, T=%d, K=%d, default r=%d", S, Tnum, K, r_default))
log_msg(sprintf("W column means: %s",
                paste(sprintf("%.3f", colMeans(W)), collapse = ", ")))


# ---------- Ground-truth parameters ------------------------------------------


# Truth coefficients for a scenario. Because B columns are centered and scaled
# to unit sample variance (B^T B = S * I_r), any unit-norm coefficient vector
# xi gives var_s(B_s^T xi) = 1. We therefore control the interaction-pattern
# amplitude directly through xi_int_scale on the eta-scale.
build_truth <- function(scenario, B, X, coords, rep = 1L,
                        kappa_true = 25,
                        alpha_true = c(0.30, -0.20),
                        beta_true = rbind(c(0.8, -0.5),   # (K-1) x (T-1)
                                          c(-0.6, 0.4)),
                        xi0_scale = 0.5,
                        xi_int_scale = 1.5) {
  r <- ncol(B); S <- nrow(B)
  Tminus <- ncol(X); Kminus <- nrow(beta_true)


  rand_unit <- function(d) { v <- rnorm(d); v / sqrt(sum(v^2)) }


  # xi_0 is a scenario-INVARIANT spatial nuisance trend: only rep (not scenario)
  # determines its draw. This ensures "null", "single_interaction", and
  # "two_interaction" differ ONLY in the xi_int terms.
  cur_state <- if (exists(".Random.seed", envir = .GlobalEnv))
    get(".Random.seed", envir = .GlobalEnv) else NULL
  set.seed(20000L + rep)
  xi0 <- sapply(seq_len(Kminus), function(k) xi0_scale * rand_unit(r))
  if (!is.null(cur_state)) assign(".Random.seed", cur_state, envir = .GlobalEnv)


  xi_int <- array(0, dim = c(r, Tminus, Kminus))
  if (scenario == "null") {
    # xi_int stays zero
  } else if (scenario == "single_interaction") {
    xi_int[, 1, 1] <- xi_int_scale * rand_unit(r)
  } else if (scenario == "two_interaction") {
    xi_int[, 1, 1] <- xi_int_scale * rand_unit(r)
    xi_int[, 2, 2] <- xi_int_scale * rand_unit(r)
  } else stop("Unknown scenario: ", scenario)


  list(alpha = alpha_true, beta = beta_true,
       xi0 = xi0, xi_int = xi_int, kappa = kappa_true)
}


# Simulate DM counts given truth.
simulate_counts <- function(truth, X, B, N) {
  S <- nrow(B); r <- ncol(B)
  Kminus <- ncol(truth$xi0); Tminus <- ncol(X)
  eta <- matrix(0, S, Kminus)
  for (k in seq_len(Kminus)) {
    eta[, k] <- truth$alpha[k] + X %*% truth$beta[k, ] + B %*% truth$xi0[, k]
    for (t in seq_len(Tminus)) {
      eta[, k] <- eta[, k] + X[, t] * as.vector(B %*% truth$xi_int[, t, k])
    }
  }
  Pi <- softmax_ref(eta)
  Kfull <- ncol(Pi)
  Y <- matrix(0L, S, Kfull)
  for (s in seq_len(S)) {
    if (N[s] == 0L) next
    a <- pmax(truth$kappa * Pi[s, ], 1e-8)
    gvec <- rgamma(Kfull, shape = a, rate = 1)
    p <- gvec / sum(gvec)
    Y[s, ] <- as.integer(rmultinom(1, size = N[s], prob = p))
  }
  list(Y = Y, Pi = Pi, eta = eta)
}


# ---------- One simulation replicate -----------------------------------------


N_BASE <- 40L
N_POIS <- 40L


one_run <- function(scenario, rep_id, r_override = NULL, seed) {
  set.seed(seed)
  B <- if (is.null(r_override)) B_default
       else build_basis_to_rank(coords, r_override)
  r <- ncol(B)


  N <- pmax(5L, N_BASE + rpois(S, N_POIS))
  truth <- build_truth(scenario, B, X_default, coords, rep = rep_id)
  sim <- simulate_counts(truth, X_default, B, N)


  t0 <- Sys.time()
  res <- tryCatch(
    compass_fit_gene(sim$Y, N, W, B, run_full = TRUE, n_starts = 3L,
                     verbose = FALSE),
    error = function(e) { message("fit failed: ", conditionMessage(e)); NULL }
  )
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))


  if (is.null(res)) {
    return(data.frame(scenario = scenario, rep_id = rep_id, r = r,
                      time_sec = dt, conv_reduced = NA_integer_,
                      conv_full = NA_integer_,
                      pi_min = NA_real_, pi_max = NA_real_,
                      Q_omni = NA_real_, df_omni = NA_integer_,
                      p_omni = NA_real_,
                      W_t1 = NA_real_, p_t1 = NA_real_,
                      W_t2 = NA_real_, p_t2 = NA_real_,
                      cond_number = NA_real_, multi_stable = NA,
                      prob_ok = NA, cond_ok = NA,
                      stringsAsFactors = FALSE))
  }


  wald_list <- res$wald$per_cell_type
  wald_Ws <- vapply(wald_list, function(x) x$W, numeric(1))
  wald_ps <- vapply(wald_list, function(x) x$p, numeric(1))


  data.frame(
    scenario = scenario, rep_id = rep_id, r = r, time_sec = dt,
    conv_reduced = res$fit_reduced$convergence,
    conv_full = res$fit_full$convergence,
    pi_min = min(res$fit_full$Pi),
    pi_max = max(res$fit_full$Pi),
    Q_omni = res$score_test$Q,
    df_omni = res$score_test$df,
    p_omni = res$score_test$p,
    W_t1 = wald_Ws[1], p_t1 = wald_ps[1],
    W_t2 = wald_Ws[2], p_t2 = wald_ps[2],
    cond_number = res$stability$cond_number,
    multi_stable = res$stability$multi_stable,
    prob_ok = res$stability$prob_ok,
    cond_ok = res$stability$cond_ok,
    stringsAsFactors = FALSE
  )
}


# ---------- Main runs --------------------------------------------------------


scenarios <- c("null", "single_interaction", "two_interaction")
N_REP <- 500L


log_msg(sprintf("--- Main runs (r = %d, n_rep per scenario = %d) ---",
                r_default, N_REP))
main_rows <- list()
for (sc in scenarios) {
  sc_ix <- match(sc, scenarios)
  for (rep_id in seq_len(N_REP)) {
    row <- one_run(sc, rep_id, seed = 10000L + 1000L * sc_ix + rep_id)
    log_msg(sprintf(
      "[%-20s rep=%d] Q=%7.2f df=%d p=%9.3g  W1=%6.2f p1=%9.3g  W2=%6.2f p2=%9.3g  conv=%s/%s  pi=[%.3g,%.3g]",
      sc, rep_id, row$Q_omni, row$df_omni, row$p_omni,
      row$W_t1, row$p_t1, row$W_t2, row$p_t2,
      row$conv_reduced, row$conv_full, row$pi_min, row$pi_max))
    main_rows[[length(main_rows) + 1L]] <- row
  }
}
main_df <- do.call(rbind, main_rows)
write.csv(main_df, file.path(OUT_DIR, "summary.csv"), row.names = FALSE)


# ---------- r-sensitivity sweep ---------------------------------------------


R_VALS <- c(2L, 4L, 8L)
N_REP_SENS <- 3L


log_msg(sprintf("--- r-sensitivity sweep (r in {%s}, n_rep=%d) ---",
                paste(R_VALS, collapse = ","), N_REP_SENS))
sens_rows <- list()
for (sc in scenarios) {
  sc_ix <- match(sc, scenarios)
  for (rv in R_VALS) {
    for (rep_id in seq_len(N_REP_SENS)) {
      row <- one_run(sc, rep_id, r_override = rv,
                     seed = 50000L + 100L * rv + 10L * sc_ix + rep_id)
      sens_rows[[length(sens_rows) + 1L]] <- row
      log_msg(sprintf("[r-sens %-20s r=%d rep=%d] Q=%7.2f p=%9.3g conv=%s/%s",
                      sc, rv, rep_id, row$Q_omni, row$p_omni,
                      row$conv_reduced, row$conv_full))
    }
  }
}
sens_df <- do.call(rbind, sens_rows)
write.csv(sens_df, file.path(OUT_DIR, "r_sensitivity.csv"), row.names = FALSE)


# ---------- Sanity checks ----------------------------------------------------


log_msg("--- Sanity checks ---")


check1 <- all(main_df$conv_reduced == 0) && all(main_df$conv_full == 0)
log_msg("[1] optimizer converges: ", check1,
        sprintf("  (reduced=%d/%d, full=%d/%d zero-codes)",
                sum(main_df$conv_reduced == 0), nrow(main_df),
                sum(main_df$conv_full == 0), nrow(main_df)))


check2 <- all(main_df$pi_min > 1e-6) && all(main_df$pi_max < 1 - 1e-6)
log_msg("[2] fitted probs in (0,1): ", check2,
        sprintf("  (min pi_min=%.3g, max pi_max=%.3g)",
                min(main_df$pi_min), max(main_df$pi_max)))


p_null <- main_df$p_omni[main_df$scenario == "null"]
p_s <- main_df$p_omni[main_df$scenario == "single_interaction"]
p_t <- main_df$p_omni[main_df$scenario == "two_interaction"]
check3 <- median(p_null) > 0.05 && median(p_s) < 0.05 && median(p_t) < 0.05
log_msg(sprintf("[3] omnibus median p: null=%.3f single=%.3g two=%.3g => %s",
                median(p_null), median(p_s), median(p_t), check3))


# Check 4: Wald block ordering
mean_W_by <- function(sc, col) mean(main_df[main_df$scenario == sc, col])
Wn1 <- mean_W_by("null", "W_t1"); Wn2 <- mean_W_by("null", "W_t2")
Ws1 <- mean_W_by("single_interaction", "W_t1")
Ws2 <- mean_W_by("single_interaction", "W_t2")
Wt1 <- mean_W_by("two_interaction", "W_t1")
Wt2 <- mean_W_by("two_interaction", "W_t2")
check4a <- Ws1 > Ws2
check4b <- Wt1 > max(Wn1, Wn2) && Wt2 > max(Wn1, Wn2)
log_msg(sprintf(
  "[4] Wald block means:  null=(%.2f,%.2f) single=(%.2f,%.2f) two=(%.2f,%.2f) | single t1>t2=%s | two active>null=%s",
  Wn1, Wn2, Ws1, Ws2, Wt1, Wt2, check4a, check4b))


# Check 5: r-sensitivity — optimizer still converges and null/alt pattern survives
conv_by_r <- aggregate(cbind(conv_full, p_omni) ~ r + scenario,
                       data = sens_df,
                       FUN = function(x) c(mean = mean(x), median = median(x)))
log_msg("[5] r-sensitivity aggregate:")
capture.output(print(conv_by_r), file = LOG_FILE, append = TRUE)
check5 <- all(vapply(R_VALS, function(rv) {
  sub <- sens_df[sens_df$r == rv, ]
  all(sub$conv_full == 0) &&
    median(sub$p_omni[sub$scenario == "null"]) > 0.05 &&
    median(sub$p_omni[sub$scenario == "single_interaction"]) < 0.10 &&
    median(sub$p_omni[sub$scenario == "two_interaction"]) < 0.10
}, logical(1)))
log_msg("[5] r-sensitivity pattern preserved: ", check5)


final_checks <- data.frame(
  check = c("optimizer_converges",
            "prob_in_0_1",
            "omnibus_null_vs_alt",
            "wald_active_dominant",
            "r_sensitivity_stable"),
  pass = c(check1, check2, check3, check4a && check4b, check5)
)
write.csv(final_checks, file.path(OUT_DIR, "sanity_checks.csv"),
          row.names = FALSE)
log_msg("--- Final check summary ---")
capture.output(print(final_checks), file = LOG_FILE, append = TRUE)
print(final_checks)


# ---------- Diagnostic plots ------------------------------------------------


png(file.path(OUT_DIR, "pvalues_hist.png"),
    width = 1050, height = 350, res = 120)
op <- par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))
for (sc in scenarios) {
  pv <- main_df$p_omni[main_df$scenario == sc]
  hist(pv, breaks = seq(0, 1, by = 0.1), xlim = c(0, 1),
       xlab = "p_omni", main = sc, col = "grey80", border = "white")
  abline(v = 0.05, col = "red", lty = 2)
}
par(op); dev.off()


png(file.path(OUT_DIR, "wald_heatmap.png"),
    width = 700, height = 400, res = 120)
mat <- rbind(
  null   = c(Wn1, Wn2),
  single = c(Ws1, Ws2),
  two    = c(Wt1, Wt2)
)
colnames(mat) <- c("t=1", "t=2")
op <- par(mar = c(4, 6, 3, 1))
image(seq_len(ncol(mat)), seq_len(nrow(mat)), t(log1p(mat)),
      axes = FALSE, xlab = "", ylab = "",
      col = hcl.colors(40, "YlOrRd", rev = TRUE))
axis(1, at = seq_len(ncol(mat)), labels = colnames(mat))
axis(2, at = seq_len(nrow(mat)), labels = rownames(mat), las = 1)
for (i in seq_len(nrow(mat))) for (j in seq_len(ncol(mat))) {
  text(j, i, sprintf("%.1f", mat[i, j]))
}
title("Mean Wald statistic (block vs scenario)")
par(op); dev.off()


png(file.path(OUT_DIR, "scenario_Q_boxplot.png"),
    width = 700, height = 400, res = 120)
op <- par(mar = c(4, 4, 2, 1))
boxplot(Q_omni ~ scenario, data = main_df,
        ylab = "Omnibus Q statistic", xlab = "", log = "y",
        col = "grey85")
par(op); dev.off()


log_msg("Artifacts written to ", OUT_DIR)


# ============================================================================
# Dump null replicates + R omnibus reference for Python cross-check.
# ============================================================================
DUMP_DIR <- file.path(OUT_DIR, "null_dump")
dir.create(DUMP_DIR, showWarnings = FALSE, recursive = TRUE)

write.table(W,         file.path(DUMP_DIR, "W.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write.table(B_default, file.path(DUMP_DIR, "B.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)

sc_ix_null <- match("null", scenarios)
r_records <- vector("list", N_REP)
for (rep_id in seq_len(N_REP)) {
  set.seed(10000L + 1000L * sc_ix_null + rep_id)
  N   <- pmax(5L, N_BASE + rpois(S, N_POIS))
  tr  <- build_truth("null", B_default, X_default, coords, rep = rep_id)
  sim <- simulate_counts(tr, X_default, B_default, N)

  write.table(sim$Y, file.path(DUMP_DIR, sprintf("Y_rep%03d.csv", rep_id)),
              sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(matrix(N, ncol = 1),
              file.path(DUMP_DIR, sprintf("N_rep%03d.csv", rep_id)),
              sep = ",", row.names = FALSE, col.names = FALSE)

  res <- compass_fit_gene(sim$Y, N, W, B_default,
                          run_full = FALSE, n_starts = 3L, verbose = FALSE)
  r_records[[rep_id]] <- data.frame(rep_id = rep_id,
                                    Q_R = res$score_test$Q,
                                    p_R = res$score_test$p)
}
write.csv(do.call(rbind, r_records),
          file.path(DUMP_DIR, "R_omnibus.csv"), row.names = FALSE)
log_msg("Wrote null dumps + R omnibus reference to ", DUMP_DIR)


log_msg("Done.")
