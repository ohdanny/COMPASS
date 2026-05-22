# =============================================================================
# Visualize the three demo scenarios in space, plus v0.1.2 cell-type-by-
# category Wald diagnostics:
#   - Cell-type composition W_s (one panel per cell type)
#   - True isoform composition pi_{sg} (one panel per category, per scenario)
#   - A representative observed Y realization (empirical proportions)
#   - Eta decomposition per scenario
#   - Null QQ plot (omnibus) and aggregate cell-type-specific power (v0.1.1)
#   - NEW v0.1.2 panels:
#       null_qq_plot_pairs.png
#       pair_power_heatmap.png
#       aggregate_vs_pair_single_interaction.png
#
# Output: PNGs written to results/20260508_demo_simulation_3scenarios_v0.1.2/viz/
# by default. Set COMPASS_DEMO_OUT_DIR to override the parent output directory.
# =============================================================================


suppressPackageStartupMessages({
  library(splines)
})

options(bitmapType = "cairo")

SCRIPT_DIR <- "./"
source(file.path(SCRIPT_DIR, "compass_core.R"))


BASE_OUT_DIR <- Sys.getenv(
  "COMPASS_DEMO_OUT_DIR",
  unset = "./Rres"
)
OUT_DIR <- file.path(BASE_OUT_DIR, "viz")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


# ---------- Rebuild the exact same setup as demo_simulation.R -----------------


set.seed(20260421)
GRID_N <- 20L; S <- GRID_N * GRID_N
coords <- as.matrix(expand.grid(u = seq(0, 1, length.out = GRID_N),
                                v = seq(0, 1, length.out = GRID_N)))


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
Tnum <- ncol(W); K <- 3L
B <- build_bspline_basis(coords, df_per_axis = 2L, orthonormalize = TRUE)
X <- W[, -Tnum, drop = FALSE]
r <- ncol(B)


# ---------- Same build_truth / simulate_counts as demo ------------------------


build_truth <- function(scenario, B, X, coords, rep = 1L,
                        kappa_true = 25,
                        alpha_true = c(0.30, -0.20),
                        beta_true = rbind(c(0.8, -0.5),
                                          c(-0.6, 0.4)),
                        xi0_scale = 0.5,
                        xi_int_scale = 1.5) {
  r <- ncol(B)
  Tminus <- ncol(X); Kminus <- nrow(beta_true)
  rand_unit <- function(d) { v <- rnorm(d); v / sqrt(sum(v^2)) }
  # xi_0 is a scenario-INVARIANT nuisance spatial trend: seed depends only on rep.
  cur_state <- if (exists(".Random.seed", envir = .GlobalEnv))
    get(".Random.seed", envir = .GlobalEnv) else NULL
  set.seed(20000L + rep)
  xi0 <- sapply(seq_len(Kminus), function(k) xi0_scale * rand_unit(r))
  if (!is.null(cur_state)) assign(".Random.seed", cur_state, envir = .GlobalEnv)
  # Interaction draws proceed under whatever RNG state the caller set, which
  # encodes both (scenario, rep).
  xi_int <- array(0, dim = c(r, Tminus, Kminus))
  if (scenario == "null") {
  } else if (scenario == "single_interaction") {
    xi_int[, 1, 1] <- xi_int_scale * rand_unit(r)
  } else if (scenario == "two_interaction") {
    xi_int[, 1, 1] <- xi_int_scale * rand_unit(r)
    xi_int[, 2, 2] <- xi_int_scale * rand_unit(r)
  }
  list(alpha = alpha_true, beta = beta_true,
       xi0 = xi0, xi_int = xi_int, kappa = kappa_true)
}


true_pi <- function(truth, X, B) {
  S <- nrow(B); Kminus <- ncol(truth$xi0); Tminus <- ncol(X)
  eta <- matrix(0, S, Kminus)
  for (k in seq_len(Kminus)) {
    eta[, k] <- truth$alpha[k] + X %*% truth$beta[k, ] + B %*% truth$xi0[, k]
    for (t in seq_len(Tminus)) {
      eta[, k] <- eta[, k] + X[, t] * as.vector(B %*% truth$xi_int[, t, k])
    }
  }
  softmax_ref(eta)
}


simulate_Y_once <- function(Pi, kappa, N) {
  S <- nrow(Pi); Kfull <- ncol(Pi)
  Y <- matrix(0L, S, Kfull)
  for (s in seq_len(S)) {
    if (N[s] == 0L) next
    a <- pmax(kappa * Pi[s, ], 1e-8)
    g <- rgamma(Kfull, shape = a, rate = 1)
    p <- g / sum(g)
    Y[s, ] <- as.integer(rmultinom(1, size = N[s], prob = p))
  }
  Y
}


# ---------- Plot helper: grid image on spot coordinates -----------------------


spot_image <- function(coords, values, main = "", zlim = c(0, 1),
                       palette = hcl.colors(50, "Viridis")) {
  # Assumes coords are on a regular grid (u, v).
  u_vals <- sort(unique(coords[, 1]))
  v_vals <- sort(unique(coords[, 2]))
  mat <- matrix(NA_real_, length(u_vals), length(v_vals))
  for (i in seq_along(u_vals)) for (j in seq_along(v_vals)) {
    idx <- which(coords[, 1] == u_vals[i] & coords[, 2] == v_vals[j])
    if (length(idx) == 1L) mat[i, j] <- values[idx]
  }
  image(u_vals, v_vals, mat, zlim = zlim, col = palette,
        xlab = "u", ylab = "v", main = main,
        useRaster = TRUE, asp = 1)
}


add_colorbar <- function(zlim, palette, title = "") {
  opar <- par(mar = c(4, 1, 3, 3))
  plot.new()
  plot.window(xlim = c(0, 1), ylim = zlim)
  ncol <- length(palette)
  y_breaks <- seq(zlim[1], zlim[2], length.out = ncol + 1L)
  rect(0, y_breaks[-length(y_breaks)], 1, y_breaks[-1L],
       col = palette, border = NA)
  axis(4, las = 1)
  title(main = title, line = 0.5, cex.main = 0.9)
  par(opar)
}


# ---------- 1) Cell-type composition ----------------------------------------


cat("Writing cell_type_composition.png\n")
ct_labels <- paste0("cell type ", seq_len(Tnum),
                    c(" (Gaussian bump near (0.25,0.25))",
                      " (Gaussian bump near (0.75,0.75))",
                      " (reference, baseline)"))
                      
png(file.path(OUT_DIR, "cell_type_composition.png"),
    width = 1400, height = 500, res = 130)
layout(matrix(c(1, 2, 3, 4), nrow = 1), widths = c(1, 1, 1, 0.25))
par(mar = c(4, 4, 3, 1))
pal <- hcl.colors(50, "Viridis")
for (t in seq_len(Tnum)) {
  spot_image(coords, W[, t], main = ct_labels[t],
             zlim = c(0, 1), palette = pal)
}
add_colorbar(c(0, 1), pal, title = "w_{st}")
dev.off()


# ---------- 2) Representative true pi per scenario --------------------------


cat("Writing scenarios_pi_true.png\n")
scenarios <- c("null", "single_interaction", "two_interaction")
scenario_pi <- list()
scenario_Y <- list()
scenario_N <- list()
for (sc in scenarios) {
  sc_ix <- match(sc, scenarios)
  set.seed(10000L + 1000L * sc_ix + 1L)  # same seed as main rep=1
  truth <- build_truth(sc, B, X, coords, rep = 1L)
  Pi_s <- true_pi(truth, X, B)
  N_s <- pmax(5L, 40L + rpois(S, 40L))
  Y_s <- simulate_Y_once(Pi_s, truth$kappa, N_s)
  scenario_pi[[sc]] <- Pi_s
  scenario_Y[[sc]] <- Y_s
  scenario_N[[sc]] <- N_s
}


isoform_labels <- c("isoform k=1", "isoform k=2",
                    "isoform k=3 (reference)")


png(file.path(OUT_DIR, "scenarios_pi_true.png"),
    width = 1400, height = 1200, res = 130)
layout(rbind(cbind(matrix(seq_len(3 * K), nrow = 3L, byrow = TRUE),
                   rep(3 * K + 1L, 3L))),
       widths = c(rep(1, K), 0.25))
par(mar = c(3.5, 3.5, 2.5, 1), oma = c(0, 3, 2.5, 0), mgp = c(2, 0.7, 0))
for (sc in scenarios) {
  Pi_s <- scenario_pi[[sc]]
  for (k in seq_len(K)) {
    main_str <- if (sc == scenarios[1]) isoform_labels[k] else ""
    spot_image(coords, Pi_s[, k], main = main_str,
               zlim = c(0, 1), palette = pal)
    if (k == 1L) mtext(sc, side = 2, line = 2.2, cex = 0.9, font = 2)
  }
}
add_colorbar(c(0, 1), pal, title = "pi_{sgk}")
mtext("True within-gene isoform composition pi_{sgk}  (rows = scenario/gene, cols = isoform)",
      side = 3, line = 0.5, outer = TRUE, cex = 1.05, font = 2)
dev.off()


# ---------- 3) Representative observed Y proportions per scenario -----------


cat("Writing scenarios_Y_observed.png\n")
png(file.path(OUT_DIR, "scenarios_Y_observed.png"),
    width = 1400, height = 1200, res = 130)
layout(rbind(cbind(matrix(seq_len(3 * K), nrow = 3L, byrow = TRUE),
                   rep(3 * K + 1L, 3L))),
       widths = c(rep(1, K), 0.25))
par(mar = c(3.5, 3.5, 2.5, 1), oma = c(0, 3, 2.5, 0), mgp = c(2, 0.7, 0))
for (sc in scenarios) {
  Y_s <- scenario_Y[[sc]]; N_s <- scenario_N[[sc]]
  Ypr <- sweep(Y_s, 1, pmax(N_s, 1), "/")
  for (k in seq_len(K)) {
    main_str <- if (sc == scenarios[1]) isoform_labels[k] else ""
    spot_image(coords, Ypr[, k], main = main_str,
               zlim = c(0, 1), palette = pal)
    if (k == 1L) mtext(sc, side = 2, line = 2.2, cex = 0.9, font = 2)
  }
}
add_colorbar(c(0, 1), pal, title = "Y_{sgk} / N_{sg}")
mtext("Observed isoform proportions Y_{sgk}/N_{sg}  (rows = scenario/gene, cols = isoform)",
      side = 3, line = 0.5, outer = TRUE, cex = 1.05, font = 2)
dev.off()


# ---------- 4) Per-scenario eta decomposition --------------------------------
# For each non-reference isoform k, visualize each additive term that builds
# up the log-ratio eta_{sgk}:
#   column 1: alpha_{gk} + X_s^T beta_{gk}     (mixture, spatially CONSTANT effect)
#   column 2: B_s^T xi_{0,gk}                  (shared spatial trend, cell-type-invariant)
#   column 3: w_{s,1} * B_s^T xi_{1,g,k}       (interaction with cell type 1)
#   column 4: w_{s,2} * B_s^T xi_{2,g,k}       (interaction with cell type 2)
#   column 5: total eta_{sgk}
# Rows are isoforms k = 1, 2 (reference k=3 has eta identically 0).


diverging <- hcl.colors(50, "Blue-Red 3")
truth_rows <- list()


compute_truth_per_scenario <- function(sc) {
  sc_ix <- match(sc, scenarios)
  set.seed(10000L + 1000L * sc_ix + 1L)  # matches rep=1
  build_truth(sc, B, X, coords, rep = 1L)
}


for (sc in scenarios) {
  truth <- compute_truth_per_scenario(sc)
  Kminus <- K - 1L; Tminus <- Tnum - 1L
  # Precompute each component S x Kminus
  mix <- matrix(0, S, Kminus); sh <- matrix(0, S, Kminus)
  int1 <- matrix(0, S, Kminus); int2 <- matrix(0, S, Kminus)
  eta_total <- matrix(0, S, Kminus)
  for (k in seq_len(Kminus)) {
    mix[, k]  <- truth$alpha[k] + X %*% truth$beta[k, ]
    sh[, k]   <- B %*% truth$xi0[, k]
    int1[, k] <- X[, 1] * as.vector(B %*% truth$xi_int[, 1, k])
    int2[, k] <- X[, 2] * as.vector(B %*% truth$xi_int[, 2, k])
    eta_total[, k] <- mix[, k] + sh[, k] + int1[, k] + int2[, k]
  }
  comps <- list(mix = mix, sh = sh, int1 = int1, int2 = int2, eta = eta_total)
  col_titles <- c(
    expression(atop(paste("cell-type mixture  ",
                          alpha[gk] + x[s]^T, beta[gk]),
                    "[same across scenarios]")),
    expression(atop(paste("shared spatial trend  ",
                          b[s]^T, xi["0,gk"]),
                    "[same across scenarios]")),
    expression(atop(paste("t=1 interaction  ",
                          w[s1], " . ", b[s]^T, xi["1,gk"]),
                    "[scenario-specific]")),
    expression(atop(paste("t=2 interaction  ",
                          w[s2], " . ", b[s]^T, xi["2,gk"]),
                    "[scenario-specific]")),
    expression(atop(paste("total  ", eta[sgk]), ""))
  )
  # Use a symmetric range per row for the diverging palette
  fn <- file.path(OUT_DIR, sprintf("eta_decomposition_%s.png", sc))
  png(fn, width = 2000, height = 950, res = 140)
  layout(rbind(cbind(matrix(seq_len(Kminus * 5L), nrow = Kminus, byrow = TRUE),
                     rep(Kminus * 5L + 1L, Kminus))),
         widths = c(rep(1, 5L), 0.25))
  par(mar = c(3.2, 3.2, 3.8, 1), oma = c(0, 4, 3, 0), mgp = c(2, 0.7, 0),
      cex.main = 0.85)
  # Per-scenario global symmetric range so rows/cols are comparable.
  all_vals <- unlist(comps)
  zmax <- max(abs(all_vals)); zlim <- c(-zmax, zmax)
  for (k in seq_len(Kminus)) {
    for (col in seq_along(comps)) {
      main_str <- if (k == 1L) col_titles[[col]] else ""
      spot_image(coords, comps[[col]][, k], main = main_str,
                 zlim = zlim, palette = diverging)
      if (col == 1L) mtext(sprintf("isoform k=%d", k),
                           side = 2, line = 2.5, cex = 0.95, font = 2)
    }
  }
  add_colorbar(zlim, diverging, title = "eta contribution")
  mtext(sprintf("Eta-decomposition for scenario: %s  (sum of columns 1-4 = column 5)", sc),
        side = 3, line = 0.5, outer = TRUE, cex = 1.1, font = 2)
  dev.off()


  # Accumulate coefficient table
  for (k in seq_len(Kminus)) {
    truth_rows[[length(truth_rows) + 1L]] <- data.frame(
      scenario = sc, isoform_k = k,
      alpha = truth$alpha[k],
      beta_t1 = truth$beta[k, 1], beta_t2 = truth$beta[k, 2],
      xi0_norm = sqrt(sum(truth$xi0[, k]^2)),
      xi_t1_norm = sqrt(sum(truth$xi_int[, 1, k]^2)),
      xi_t2_norm = sqrt(sum(truth$xi_int[, 2, k]^2)),
      kappa = truth$kappa,
      stringsAsFactors = FALSE
    )
  }
}
coef_tbl <- do.call(rbind, truth_rows)
write.csv(coef_tbl, file.path(OUT_DIR, "truth_coefficients.csv"), row.names = FALSE)


# ---------- 5) Null QQ plot and cell-type-specific power ----------------------


SUMMARY_FILE <- file.path(BASE_OUT_DIR, "summary.csv")
POWER_FILE <- file.path(OUT_DIR, "cell_type_specific_power.csv")
ALPHA <- 0.05


if (file.exists(SUMMARY_FILE)) {
  cat("Writing null_qq_plot.png\n")
  sim_summary <- read.csv(SUMMARY_FILE, stringsAsFactors = FALSE)


  p_null <- sim_summary$p_omni[sim_summary$scenario == "null"]
  p_null <- sort(p_null[is.finite(p_null) & !is.na(p_null)])
  if (length(p_null) > 0L) {
    exp_p <- ppoints(length(p_null))
    png(file.path(OUT_DIR, "null_qq_plot.png"),
        width = 650, height = 650, res = 130)
    par(mar = c(4.5, 4.5, 3, 1))
    plot(-log10(exp_p), -log10(p_null),
         xlab = "Expected -log10(p) under null",
         ylab = "Observed -log10(p)",
         main = "Null scenario QQ plot",
         pch = 19, col = "grey30")
    abline(0, 1, col = "red", lty = 2, lwd = 2)
    dev.off()
  } else {
    warning("No finite null p-values available for QQ plot.")
  }


  cat("Writing cell_type_specific_power.csv and cell_type_specific_power.png\n")
  power_rows <- list()
  for (sc in c("single_interaction", "two_interaction")) {
    sub <- sim_summary[sim_summary$scenario == sc, ]
    active <- list(
      t1 = sc %in% c("single_interaction", "two_interaction"),
      t2 = sc == "two_interaction"
    )
    for (cell_type in c("t1", "t2")) {
      p_col <- paste0("p_", cell_type)
      p_vals <- sub[[p_col]]
      p_vals <- p_vals[is.finite(p_vals) & !is.na(p_vals)]
      power_rows[[length(power_rows) + 1L]] <- data.frame(
        scenario = sc,
        cell_type = cell_type,
        p_value_column = p_col,
        active_truth = active[[cell_type]],
        n_rep = length(p_vals),
        alpha = ALPHA,
        power = mean(p_vals < ALPHA),
        stringsAsFactors = FALSE
      )
    }
  }
  power_tbl <- do.call(rbind, power_rows)
  write.csv(power_tbl, POWER_FILE, row.names = FALSE)


  png(file.path(OUT_DIR, "cell_type_specific_power.png"),
      width = 850, height = 500, res = 130)
  op <- par(mar = c(6, 4.5, 3, 1))
  bar_cols <- ifelse(power_tbl$active_truth, "#3B7EA1", "grey75")
  bp <- barplot(power_tbl$power, ylim = c(0, 1),
                names.arg = paste(power_tbl$scenario, power_tbl$cell_type,
                                  sep = "\n"),
                las = 2, col = bar_cols, border = "white",
                ylab = sprintf("Pr(p < %.2f)", ALPHA),
                main = "Cell-type-specific interaction test power")
  abline(h = ALPHA, col = "red", lty = 2)
  text(bp, pmin(power_tbl$power + 0.05, 0.98),
       labels = sprintf("%.2f", power_tbl$power), cex = 0.85)
  legend("topright", fill = c("#3B7EA1", "grey75"), border = "white",
         legend = c("active truth", "inactive truth"), bty = "n")
  par(op)
  dev.off()
} else {
  warning("No simulation summary found at ", SUMMARY_FILE,
          "; skipping QQ plot and power visualizations.")
}


# ---------- 6) Cell-type-by-category Wald diagnostics (NEW for v0.1.2) -------


PAIR_POWER_FILE <- file.path(BASE_OUT_DIR, "pair_power_summary.csv")
pair_cols <- c("p_t1k1", "p_t1k2", "p_t2k1", "p_t2k2")
holm_cols <- c("p_holm_t1k1", "p_holm_t1k2", "p_holm_t2k1", "p_holm_t2k2")


if (file.exists(SUMMARY_FILE) && file.exists(PAIR_POWER_FILE) &&
    all(pair_cols %in% colnames(sim_summary))) {


  # (a) Pooled null QQ over all 4 pairs.
  cat("Writing null_qq_plot_pairs.png\n")
  p_null_pairs <- unlist(lapply(pair_cols, function(co) {
    sim_summary[[co]][sim_summary$scenario == "null"]
  }))
  p_null_pairs <- sort(p_null_pairs[is.finite(p_null_pairs)])
  if (length(p_null_pairs) > 0L) {
    exp_p <- ppoints(length(p_null_pairs))
    png(file.path(OUT_DIR, "null_qq_plot_pairs.png"),
        width = 700, height = 700, res = 130)
    par(mar = c(4.5, 4.5, 3, 1))
    plot(-log10(exp_p), -log10(p_null_pairs),
         xlab = "Expected -log10(p) under null",
         ylab = "Observed -log10(p)",
         main = sprintf(
           "Null QQ: cell-type x category Wald (%d pooled p-values)",
           length(p_null_pairs)),
         pch = 19, col = "grey30")
    abline(0, 1, col = "red", lty = 2, lwd = 2)
    dev.off()
  } else {
    warning("No finite pair p-values under null; skipping null_qq_plot_pairs.png.")
  }


  # (b) Pair power heatmap (raw + Holm, with ground-truth markers).
  cat("Writing pair_power_heatmap.png\n")
  pair_power_tbl <- read.csv(PAIR_POWER_FILE, stringsAsFactors = FALSE)
  pair_power_tbl$tk <- sprintf("(%d,%d)", pair_power_tbl$t, pair_power_tbl$k)
  pair_levels <- c("(1,1)", "(1,2)", "(2,1)", "(2,2)")
  scen_levels <- c("null", "single_interaction", "two_interaction")


  to_matrix <- function(col) {
    M <- matrix(NA_real_, length(scen_levels), length(pair_levels),
                dimnames = list(scen_levels, pair_levels))
    for (i in seq_len(nrow(pair_power_tbl))) {
      sc <- pair_power_tbl$scenario[i]; tk <- pair_power_tbl$tk[i]
      if (sc %in% scen_levels && tk %in% pair_levels) M[sc, tk] <- pair_power_tbl[[col]][i]
    }
    M
  }
  M_raw <- to_matrix("rate_raw")
  M_holm <- to_matrix("rate_holm")
  M_truth <- to_matrix("active_truth") > 0


  pal_v <- hcl.colors(50, "Viridis")
  png(file.path(OUT_DIR, "pair_power_heatmap.png"),
      width = 1400, height = 600, res = 140)
  layout(matrix(c(1, 2, 3), nrow = 1), widths = c(1, 1, 0.22))
  par(mar = c(5, 8, 3.5, 1), mgp = c(2.4, 0.7, 0))
  for (mat_info in list(list(M = M_raw,  title = "Pair power (raw, alpha=0.05)"),
                        list(M = M_holm, title = "Pair power (Holm-adjusted)"))) {
    M <- mat_info$M
    image(seq_len(ncol(M)), seq_len(nrow(M)),
          t(M[rev(seq_len(nrow(M))), , drop = FALSE]),
          zlim = c(0, 1), col = pal_v, axes = FALSE,
          xlab = "(t, k) pair", ylab = "",
          main = mat_info$title)
    axis(1, at = seq_len(ncol(M)), labels = colnames(M), las = 1)
    axis(2, at = seq_len(nrow(M)), labels = rev(rownames(M)), las = 1)
    for (i in seq_len(nrow(M))) for (j in seq_len(ncol(M))) {
      val <- M[i, j]; truth <- M_truth[i, j]
      if (is.na(val)) next
      lab <- if (truth) sprintf("%.2f *", val) else sprintf("%.2f", val)
      ycoord <- nrow(M) - i + 1L
      text(j, ycoord, lab,
           col = ifelse(val > 0.5, "white", "black"), cex = 0.95)
    }
  }
  add_colorbar(c(0, 1), pal_v, title = "Pr(reject)")
  dev.off()


  # (c) Aggregate vs pair test, side-by-side, on single_interaction.
  # The narrative: under single_interaction the only active pair is (1,1),
  # so wald_tests() correctly rejects t=1 in aggregate but cannot localize
  # within k; wald_tests_by_category() should reject (1,1) only.
  cat("Writing aggregate_vs_pair_single_interaction.png\n")
  sub_si <- sim_summary[sim_summary$scenario == "single_interaction", ]
  if (nrow(sub_si) > 0L) {
    agg_rates <- c(t1 = mean(sub_si$p_t1 < ALPHA, na.rm = TRUE),
                   t2 = mean(sub_si$p_t2 < ALPHA, na.rm = TRUE))
    agg_truth <- c(TRUE, FALSE)
    pair_rates <- vapply(pair_cols, function(co)
      mean(sub_si[[co]] < ALPHA, na.rm = TRUE), numeric(1))
    pair_truth <- c(TRUE, FALSE, FALSE, FALSE)
    pair_labels <- c("(1,1)", "(1,2)", "(2,1)", "(2,2)")


    png(file.path(OUT_DIR, "aggregate_vs_pair_single_interaction.png"),
        width = 1400, height = 600, res = 140)
    par(mfrow = c(1, 2), mar = c(5.5, 4.5, 4, 1), mgp = c(2.5, 0.7, 0))


    bp1 <- barplot(agg_rates, ylim = c(0, 1),
                   col = ifelse(agg_truth, "#3B7EA1", "grey75"),
                   border = "white",
                   main = "wald_tests() per cell type\n(df = r * (K-1), aggregate over k)",
                   ylab = sprintf("Pr(p < %.2f)", ALPHA),
                   names.arg = c("t=1", "t=2"))
    abline(h = ALPHA, col = "red", lty = 2)
    text(bp1, pmin(agg_rates + 0.04, 0.97),
         labels = sprintf("%.2f", agg_rates), cex = 0.95)


    bp2 <- barplot(pair_rates, ylim = c(0, 1),
                   col = ifelse(pair_truth, "#3B7EA1", "grey75"),
                   border = "white",
                   main = "wald_tests_by_category() per (t, k)\n(df = r, NEW in v0.1.2)",
                   ylab = sprintf("Pr(p < %.2f)", ALPHA),
                   names.arg = pair_labels)
    abline(h = ALPHA, col = "red", lty = 2)
    text(bp2, pmin(pair_rates + 0.04, 0.97),
         labels = sprintf("%.2f", pair_rates), cex = 0.95)
    legend("topright", fill = c("#3B7EA1", "grey75"), border = "white",
           legend = c("active truth", "inactive truth"), bty = "n")
    mtext("scenario: single_interaction (only (t=1, k=1) active)",
          side = 3, line = -1.4, outer = TRUE, cex = 1.05, font = 2)
    dev.off()
  } else {
    warning("No single_interaction reps; skipping aggregate_vs_pair plot.")
  }
} else {
  warning("Pair-level columns or pair_power_summary.csv missing; skipping ",
          "v0.1.2 cell-type-by-category visualizations.")
}


cat(sprintf("\nSimulation dimensions:\n"))
cat(sprintf("  spots S               = %d (%dx%d grid)\n", S, GRID_N, GRID_N))
cat(sprintf("  cell types T          = %d (reference = last)\n", Tnum))
cat(sprintf("  isoforms per gene K_g = %d (reference = last)\n", K))
cat(sprintf("  genes (= scenarios)   = %d  [null, single_interaction, two_interaction]\n",
            length(scenarios)))
cat(sprintf("  spatial basis rank r  = %d\n", r))
cat("\nTruth coefficient norms per (scenario, isoform):\n")
print(coef_tbl, row.names = FALSE, digits = 3)
cat(sprintf("\nArtifacts:\n  %s\n", file.path(OUT_DIR, "cell_type_composition.png")))
cat(sprintf("  %s\n", file.path(OUT_DIR, "scenarios_pi_true.png")))
cat(sprintf("  %s\n", file.path(OUT_DIR, "scenarios_Y_observed.png")))
for (sc in scenarios) {
  cat(sprintf("  %s\n", file.path(OUT_DIR, sprintf("eta_decomposition_%s.png", sc))))
}
cat(sprintf("  %s\n", file.path(OUT_DIR, "truth_coefficients.csv")))
if (file.exists(SUMMARY_FILE)) {
  cat(sprintf("  %s\n", file.path(OUT_DIR, "null_qq_plot.png")))
  cat(sprintf("  %s\n", file.path(OUT_DIR, "cell_type_specific_power.png")))
  cat(sprintf("  %s\n", POWER_FILE))
}
if (file.exists(SUMMARY_FILE) && file.exists(PAIR_POWER_FILE)) {
  cat(sprintf("  %s\n", file.path(OUT_DIR, "null_qq_plot_pairs.png")))
  cat(sprintf("  %s\n", file.path(OUT_DIR, "pair_power_heatmap.png")))
  cat(sprintf("  %s\n",
              file.path(OUT_DIR, "aggregate_vs_pair_single_interaction.png")))
  cat(sprintf("  %s\n", PAIR_POWER_FILE))
}
