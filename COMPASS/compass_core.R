# =============================================================================
# COMPASS core functions  ---  v0.1.2
#
# Companion to documentation/model_v0.1.2.tex.
#
# Changes from v0.1.1:
#   * Added wald_tests_by_category(): cell-type-by-category Wald test from
#     model_v0.1.2.tex section "Cell-type-by-category Wald tests". For each
#     non-reference cell type t in {1,..,T-1} and non-reference transcript-
#     processing category k in {1,..,K-1}, computes the per-pair statistic
#         W_{tgk} = xi_hat_{tgk}^T Sigma_{tgk}^{-1} xi_hat_{tgk}  ~  chi^2_r,
#     where Sigma_{tgk} is the (xi_int[[k]][[t]], xi_int[[k]][[t]]) block of
#     the inverse observed information matrix at the full-model MLE.
#   * Pair-specific eligibility gate: a (t, k) pair is reportable iff
#         E_{tg} = sum_s N_s w_{st} >= E_min,
#         M_{tg} = sum_s 1{N_s>0, w_{st} >= tau_w} >= M_min,
#         C_{gk} = sum_s Y_{sk} >= C_min.
#     Ineligible pairs are not tested (W_tgk and p_tgk left NA).
#   * Holm adjustment is applied across the eligible pairs only, within gene.
#   * Reuses cov_full from wald_tests() so compass_fit_gene() inverts the
#     full-model observed information matrix exactly once per gene.
#
# Changes from v0.1.0:
#   * Added neg_loglik_grad(): closed-form analytic gradient of the negative
#     Dirichlet-multinomial log-likelihood with respect to (vec(Gamma), log_kappa).
#     Verified to match numDeriv::grad(method="Richardson") to its own
#     truncation-error floor across a range of dimensions (see
#     compass_self_test_gradient() at the bottom of this file).
#   * fit_model() now passes the analytic gradient as gr=  to optim(),
#     replacing the per-iteration finite-difference gradient (the previous
#     dominant cost for BFGS on the full-model 271-parameter problem).
#   * omnibus_score_test() now uses the analytic gradient directly in place of
#     numDeriv::grad and computes the Hessian via numDeriv::jacobian on the
#     analytic gradient, dropping the cost from O(p^2) NLL evals to O(p)
#     gradient evals.
#   * wald_tests() and check_stability() likewise compute Hessians via
#     jacobian-of-analytic-gradient.
#
# Mathematical derivation of the analytic gradient (per gene g, suppressed):
#
#   eta = Z * Gamma  (S x (K-1)),   tilde_eta_s = (eta_s, 0)  in R^K
#   pi_{s.} = softmax(tilde_eta_s)                              (rows sum to 1)
#   alpha_{sk} = kappa * pi_{sk}                                (a0 := kappa)
#   ell_s = log Gamma(kappa) - log Gamma(N_s + kappa)
#         + sum_{k=1..K} [ log Gamma(Y_{sk} + alpha_{sk}) - log Gamma(alpha_{sk}) ]
#
# Step A -- d ell_s / d pi_{sk} (with kappa fixed, treating pi as K-vector):
#   g_{sk} = kappa * [ digamma(Y_{sk} + kappa pi_{sk})
#                      - digamma(kappa pi_{sk}) ]      (k = 1..K)
# Note g_{sk} = 0 whenever Y_{sk} = 0.
#
# Step B -- chain rule through softmax with reference category K fixed:
#   For k in {1..K-1} and j in {1..K},  d pi_{sj}/d eta_{sk} = pi_{sj}(d_{jk} - pi_{sk}).
#   Let h_{sj} = pi_{sj} g_{sj},  hbar_s = sum_j h_{sj}.   Then
#       R_{sk} := d ell_s / d eta_{sk} = h_{sk} - pi_{sk} * hbar_s,   k = 1..K-1.
#
# Step C -- d ell / d Gamma:
#   eta_{sk} = z_s^T gamma_k  =>  d ell / d gamma_k = sum_s z_s R_{sk}
#                              =>  grad_Gamma = Z^T R    (p x (K-1))
#
# Step D -- d ell / d kappa  (pi independent of kappa):
#   d ell_s / d kappa = digamma(kappa) - digamma(N_s + kappa)
#                       + sum_{k=1..K} pi_{sk} [ digamma(Y_{sk}+alpha_{sk})
#                                                - digamma(alpha_{sk}) ]
#                     = digamma(kappa) - digamma(N_s + kappa) + hbar_s / kappa
#   With zeta = log(kappa), d kappa / d zeta = kappa, so
#     d ell / d zeta = kappa * [ S*digamma(kappa) - sum_s digamma(N_s+kappa) ]
#                    + sum_s hbar_s.
#
# For neg_loglik = - ell, both are negated.
# =============================================================================


if (!requireNamespace("numDeriv", quietly = TRUE)) {
  install.packages("numDeriv", repos = "https://cloud.r-project.org")
}


suppressPackageStartupMessages({
  library(numDeriv)
  library(splines)
})


COMPASS_CORE_VERSION <- "0.1.2"


# ---------- Parameter indexing ----------------------------------------------


param_index <- function(K, Tnum, r, full = TRUE) {
  Kminus <- K - 1L
  Tminus <- Tnum - 1L
  p_per <- if (full) 1L + Tminus + r * Tnum else 1L + Tminus + r
  offs <- (seq_len(Kminus) - 1L) * p_per


  alpha_idx <- as.integer(offs + 1L)
  beta_idx <- if (Tminus > 0L) {
    lapply(offs, function(o) as.integer(o + 1L + seq_len(Tminus)))
  } else {
    rep(list(integer(0)), Kminus)
  }
  xi0_idx <- lapply(offs, function(o) as.integer(o + (1L + Tminus) + seq_len(r)))


  xi_int_idx <- if (full && Tminus > 0L) {
    lapply(seq_len(Kminus), function(k) {
      o <- offs[k]
      lapply(seq_len(Tminus), function(t) {
        as.integer(o + (1L + Tminus + r) + ((t - 1L) * r + seq_len(r)))
      })
    })
  } else NULL


  kappa_idx <- as.integer(Kminus * p_per + 1L)
  total <- kappa_idx


  list(K = K, Tnum = Tnum, r = r, full = full,
       p_per = p_per, total = total,
       alpha = alpha_idx, beta = beta_idx, xi0 = xi0_idx,
       xi_int = xi_int_idx, kappa = kappa_idx)
}


xi_flat <- function(idx) {
  if (is.null(idx$xi_int)) return(integer(0))
  as.integer(unlist(lapply(idx$xi_int, function(xk) unlist(xk))))
}


# ---------- Design matrix ----------------------------------------------------


build_design <- function(X, B, full = TRUE) {
  S <- nrow(B); r <- ncol(B)
  if (is.null(X) || ncol(X) == 0L) {
    return(cbind(rep(1, S), B))
  }
  Z_red <- cbind(rep(1, S), X, B)
  if (!full) return(Z_red)
  Tminus <- ncol(X)
  inter <- do.call(cbind, lapply(seq_len(Tminus), function(t) X[, t] * B))
  cbind(Z_red, inter)
}


# ---------- Spatial basis (tensor-product cubic B-spline) --------------------


build_bspline_basis <- function(coords, df_per_axis = 2L,
                                orthonormalize = TRUE,
                                unit_variance = TRUE) {
  stopifnot(ncol(coords) == 2L)
  u <- coords[, 1]; v <- coords[, 2]
  df_per_axis <- as.integer(df_per_axis)
  if (df_per_axis < 3L) {
    Bu <- splines::bs(u, df = df_per_axis, degree = df_per_axis,
                      intercept = FALSE)
    Bv <- splines::bs(v, df = df_per_axis, degree = df_per_axis,
                      intercept = FALSE)
  } else {
    Bu <- splines::bs(u, df = df_per_axis, intercept = FALSE)
    Bv <- splines::bs(v, df = df_per_axis, intercept = FALSE)
  }
  d1 <- ncol(Bu); d2 <- ncol(Bv)
  B_raw <- matrix(0, nrow(coords), d1 * d2)
  col <- 0L
  for (j in seq_len(d2)) for (i in seq_len(d1)) {
    col <- col + 1L
    B_raw[, col] <- Bu[, i] * Bv[, j]
  }
  B_c <- scale(B_raw, center = TRUE, scale = FALSE)
  attr(B_c, "scaled:center") <- NULL
  if (orthonormalize) {
    qr_obj <- qr(B_c)
    rk <- qr_obj$rank
    B_out <- qr.Q(qr_obj)[, seq_len(rk), drop = FALSE]
    if (unit_variance) B_out <- B_out * sqrt(nrow(coords))
  } else {
    B_out <- B_c
  }
  attr(B_out, "df_per_axis") <- df_per_axis
  B_out
}


build_basis_to_rank <- function(coords, r_target) {
  df_axis <- max(2L, as.integer(ceiling(sqrt(r_target))))
  while (TRUE) {
    B <- build_bspline_basis(coords, df_per_axis = df_axis,
                             orthonormalize = TRUE)
    if (ncol(B) >= r_target || df_axis >= 6L) break
    df_axis <- df_axis + 1L
  }
  B[, seq_len(min(r_target, ncol(B))), drop = FALSE]
}


# ---------- Softmax & Dirichlet-multinomial ---------------------------------


softmax_ref <- function(eta) {
  eta_full <- cbind(eta, 0)
  m <- apply(eta_full, 1, max)
  e <- exp(eta_full - m)
  e / rowSums(e)
}


dm_loglik <- function(Y, N, Pi, kappa, include_const = TRUE) {
  alpha <- kappa * Pi
  a0 <- rowSums(alpha)
  ll <- lgamma(a0) - lgamma(N + a0) +
    rowSums(lgamma(Y + alpha) - lgamma(alpha))
  if (include_const) {
    ll <- ll + lgamma(N + 1) - rowSums(lgamma(Y + 1))
  }
  sum(ll)
}


# ---------- Negative log-likelihood (forward) -------------------------------


neg_loglik <- function(params, Y, N, Z, K) {
  p_per <- ncol(Z)
  Kminus <- K - 1L
  if (length(params) != Kminus * p_per + 1L) {
    stop("neg_loglik: length(params) mismatch")
  }
  gammas <- params[seq_len(Kminus * p_per)]
  log_kappa <- params[Kminus * p_per + 1L]
  kappa <- exp(log_kappa)
  if (!is.finite(kappa) || kappa <= 0 || kappa > 1e8) return(1e12)
  Gamma <- matrix(gammas, nrow = p_per, ncol = Kminus)
  eta <- Z %*% Gamma
  if (any(!is.finite(eta))) return(1e12)
  eta <- pmin(pmax(eta, -40), 40)
  Pi <- softmax_ref(eta)
  val <- -dm_loglik(Y, N, Pi, kappa)
  if (!is.finite(val)) return(1e12)
  val
}


# ---------- Negative log-likelihood gradient (analytic, NEW in v0.1.1) ------
#
# Returns the gradient of neg_loglik at `params`, in the same flat layout that
# neg_loglik consumes:  c(as.vector(d/dGamma), d/dlog_kappa).
#
# Mirrors neg_loglik's eta-clipping: where the forward pass clipped eta to
# +/- 40, the chain-rule factor d eta_clipped / d eta is 0, so the gradient
# contribution from those entries is zeroed (the forward pass and gradient
# remain exactly consistent at all parameter values).
#
# In bad regimes (kappa out of range, non-finite eta) the forward pass returns
# the penalty value 1e12; here we return a zero gradient so that BFGS will
# shrink its line-search step rather than propose an even more pathological
# move along an arbitrary direction.


neg_loglik_grad <- function(params, Y, N, Z, K) {
  p_per <- ncol(Z)
  Kminus <- K - 1L
  S <- nrow(Y)
  if (length(params) != Kminus * p_per + 1L) {
    stop("neg_loglik_grad: length(params) mismatch")
  }
  total <- Kminus * p_per + 1L


  gammas <- params[seq_len(Kminus * p_per)]
  log_kappa <- params[total]
  kappa <- exp(log_kappa)
  if (!is.finite(kappa) || kappa <= 0 || kappa > 1e8) return(rep(0, total))


  Gamma <- matrix(gammas, nrow = p_per, ncol = Kminus)
  eta <- Z %*% Gamma
  if (any(!is.finite(eta))) return(rep(0, total))


  eta_clipped <- pmin(pmax(eta, -40), 40)
  not_clipped <- (eta == eta_clipped)                   # S x (K-1) logical


  Pi <- softmax_ref(eta_clipped)                        # S x K
  alpha <- kappa * Pi                                   # S x K
  # g_{sk} = kappa * [ digamma(Y + kappa pi) - digamma(kappa pi) ]
  G <- kappa * (digamma(Y + alpha) - digamma(alpha))    # S x K
  H <- Pi * G                                           # h_{sk}, S x K
  hbar <- rowSums(H)                                    # length S


  # R_{sk} = h_{sk} - pi_{sk} * hbar  for k = 1..K-1
  Pi_red <- Pi[, seq_len(Kminus), drop = FALSE]
  R <- H[, seq_len(Kminus), drop = FALSE] - Pi_red * hbar
  R[!not_clipped] <- 0                                  # zero across clipped eta


  # d ell / d Gamma (positive sign); negate for neg_loglik.
  grad_gamma <- -crossprod(Z, R)                        # p_per x (K-1)


  # d ell / d log_kappa
  dell_dlogk <- kappa * (S * digamma(kappa) - sum(digamma(N + kappa))) + sum(hbar)
  grad_logk <- -dell_dlogk


  c(as.vector(grad_gamma), grad_logk)
}


# ---------- Multi-start MLE with analytic gradient --------------------------


fit_model <- function(Y, N, X, B, K,
                      full = TRUE, n_starts = 3L,
                      init_scale = 0.1, method = "BFGS", maxit = 500L,
                      init_params = NULL, verbose = FALSE, seed = NULL,
                      use_analytic_grad = TRUE) {
  Tnum <- if (is.null(X)) 1L else ncol(X) + 1L
  r <- ncol(B)
  Kminus <- K - 1L
  Z <- build_design(X, B, full = full)
  p_per <- ncol(Z)
  total <- Kminus * p_per + 1L


  if (!is.null(seed)) set.seed(seed)


  fn <- function(p) neg_loglik(p, Y, N, Z, K)
  gr <- if (isTRUE(use_analytic_grad)) {
    function(p) neg_loglik_grad(p, Y, N, Z, K)
  } else NULL


  starts <- vector("list", n_starts)
  best <- NULL
  for (s in seq_len(n_starts)) {
    if (s == 1L && !is.null(init_params) && length(init_params) == total) {
      p0 <- init_params
    } else if (s == 1L) {
      p0 <- rep(0, total)
      p0[total] <- log(5)
    } else {
      p0 <- rnorm(total, 0, init_scale)
      p0[total] <- log(5) + rnorm(1, 0, 0.2)
    }
    fit <- tryCatch(
      optim(p0, fn, gr = gr,
            method = method,
            control = list(maxit = maxit, reltol = 1e-8)),
      error = function(e) {
        if (verbose) message("optim error: ", conditionMessage(e))
        NULL
      }
    )
    starts[[s]] <- fit
    if (!is.null(fit) && (is.null(best) || fit$value < best$value)) best <- fit
  }
  if (is.null(best)) stop("All optim starts failed.")


  gammas <- best$par[seq_len(Kminus * p_per)]
  log_kappa <- best$par[total]
  Gamma <- matrix(gammas, nrow = p_per, ncol = Kminus)
  eta <- Z %*% Gamma
  eta <- pmin(pmax(eta, -40), 40)
  Pi <- softmax_ref(eta)


  list(par = best$par, value = best$value,
       loglik = -best$value,
       convergence = best$convergence,
       Gamma = Gamma, kappa = exp(log_kappa), log_kappa = log_kappa,
       Pi = Pi, eta = eta,
       Z = Z, Y = Y, N = N, X = X, B = B, K = K,
       Tnum = Tnum, r = r, full = full,
       p_per = p_per, total = total,
       starts = starts)
}


# ---------- Helper: Hessian via jacobian-of-gradient ------------------------


# Computes the Hessian of neg_loglik at `params` by taking the Jacobian of
# the analytic gradient. Costs O(p) gradient evaluations rather than the
# O(p^2) NLL evaluations that numDeriv::hessian would require, and inherits
# the analytic gradient's per-evaluation O(S p K) cost.
hessian_via_grad <- function(params, Y, N, Z, K, method = "Richardson") {
  gr <- function(p) neg_loglik_grad(p, Y, N, Z, K)
  J <- numDeriv::jacobian(gr, params, method = method)
  (J + t(J)) / 2  # symmetrize numerical noise
}


# ---------- Omnibus score test (under reduced-model fit) ---------------------


omnibus_score_test <- function(Y, N, X, B, K,
                               fit_reduced = NULL,
                               n_starts = 3L,
                               regularize = 1e-6, ...) {
  Tnum <- ncol(X) + 1L
  r <- ncol(B)
  Kminus <- K - 1L


  if (is.null(fit_reduced)) {
    fit_reduced <- fit_model(Y, N, X, B, K, full = FALSE,
                             n_starts = n_starts, ...)
  }


  idx_full <- param_index(K, Tnum, r, full = TRUE)
  idx_red <- param_index(K, Tnum, r, full = FALSE)


  # Lift reduced-fit parameters into the full parameter vector at Xi = 0.
  params_full <- rep(0, idx_full$total)
  for (k in seq_len(Kminus)) {
    params_full[idx_full$alpha[k]] <- fit_reduced$par[idx_red$alpha[k]]
    if (length(idx_full$beta[[k]]) > 0L) {
      params_full[idx_full$beta[[k]]] <- fit_reduced$par[idx_red$beta[[k]]]
    }
    params_full[idx_full$xi0[[k]]] <- fit_reduced$par[idx_red$xi0[[k]]]
  }
  params_full[idx_full$kappa] <- fit_reduced$par[idx_red$kappa]


  Z_full <- build_design(X, B, full = TRUE)


  # Analytic gradient (= -score); Hessian via jacobian-of-gradient.
  g <- neg_loglik_grad(params_full, Y, N, Z_full, K)
  H <- hessian_via_grad(params_full, Y, N, Z_full, K)


  xi_idx <- xi_flat(idx_full)
  theta_idx <- setdiff(seq_len(idx_full$total), xi_idx)


  U <- -g[xi_idx]                                    # score = -grad(NLL)
  I_XX <- H[xi_idx, xi_idx, drop = FALSE]
  I_Xt <- H[xi_idx, theta_idx, drop = FALSE]
  I_tt <- H[theta_idx, theta_idx, drop = FALSE]


  I_tt_reg <- I_tt + diag(regularize, nrow(I_tt))
  J <- I_XX - I_Xt %*% solve(I_tt_reg, t(I_Xt))
  J <- (J + t(J)) / 2


  eig <- eigen(J, symmetric = TRUE)
  pos_floor <- max(1e-8, regularize * max(abs(eig$values)))
  eig_vals <- pmax(eig$values, pos_floor)
  J_inv <- eig$vectors %*% (t(eig$vectors) / eig_vals)


  Q <- as.numeric(crossprod(U, J_inv %*% U))
  df <- length(xi_idx)
  pval <- pchisq(Q, df = df, lower.tail = FALSE)


  list(Q = Q, df = df, p = pval, U = U, J = J,
       I_XX = I_XX, I_Xt = I_Xt, I_tt = I_tt,
       eig_values = eig$values,
       fit_reduced = fit_reduced)
}


# ---------- Cell-type-specific Wald tests on the full-model fit --------------


wald_tests <- function(fit_full, regularize = 1e-6, hessian = NULL) {
  stopifnot(isTRUE(fit_full$full))
  K <- fit_full$K; Tnum <- fit_full$Tnum; r <- fit_full$r
  Kminus <- K - 1L; Tminus <- Tnum - 1L


  idx <- param_index(K, Tnum, r, full = TRUE)
  Y <- fit_full$Y; N <- fit_full$N; Z <- fit_full$Z


  if (is.null(hessian)) {
    H <- hessian_via_grad(fit_full$par, Y, N, Z, K)
  } else {
    H <- hessian
  }
  H_reg <- H + diag(regularize, nrow(H))
  V <- tryCatch(solve(H_reg), error = function(e) MASS::ginv(H_reg))


  out <- vector("list", Tminus)
  for (t in seq_len(Tminus)) {
    blk_idx <- unlist(lapply(seq_len(Kminus), function(k) idx$xi_int[[k]][[t]]))
    Xi_hat <- fit_full$par[blk_idx]
    V_blk <- V[blk_idx, blk_idx, drop = FALSE]
    V_blk <- (V_blk + t(V_blk)) / 2
    V_blk_reg <- V_blk + diag(regularize, nrow(V_blk))
    W <- as.numeric(crossprod(Xi_hat, solve(V_blk_reg, Xi_hat)))
    df_t <- length(blk_idx)
    p_t <- pchisq(W, df = df_t, lower.tail = FALSE)
    out[[t]] <- list(t = t, W = W, df = df_t, p = p_t,
                     Xi_hat = Xi_hat, V_blk = V_blk)
  }
  ps <- vapply(out, function(x) x$p, numeric(1))
  p_holm <- p.adjust(ps, method = "holm")
  for (t in seq_len(Tminus)) out[[t]]$p_holm <- p_holm[t]


  list(per_cell_type = out, I_full = H, cov_full = V)
}


# ---------- Cell-type-by-category Wald tests (NEW in v0.1.2) -----------------
#
# Implements the per-(t, k) Wald test from model_v0.1.2.tex section
# "Cell-type-by-category Wald tests".
#
# For non-reference cell type t in {1,..,T-1} and non-reference category
# k in {1,..,K-1}, the statistic is
#     W_{tgk} = xi_hat_{tgk}^T Sigma_{tgk}^{-1} xi_hat_{tgk}  ~  chi^2_r,
# where Sigma_{tgk} is the (idx$xi_int[[k]][[t]], idx$xi_int[[k]][[t]])
# block of the inverse observed information matrix at the full-model MLE.
# This is the same per-block extraction pattern as wald_tests(), but indexed
# per pair (t, k) instead of stacking all k for a given t.
#
# Eligibility: a pair (t, k) is reportable iff
#     E_{tg} = sum_s N_s w_{st} >= E_min,
#     M_{tg} = sum_s 1{N_s > 0, w_{st} >= tau_w} >= M_min,
#     C_{gk} = sum_s Y_{sk} >= C_min.
# Per the methodology, ineligible pairs are *not* tested: W_tgk and p_tgk
# are left NA in the summary, and Holm adjustment is performed across the
# eligible pairs only.
#
# Hessian / covariance reuse: the cell-type-aggregate wald_tests() already
# computes V = solve(H + reg*I) for the full-model observed information.
# Pass that V here as `cov_full` to skip the second inversion. Otherwise,
# `hessian` is inverted; otherwise the Hessian is recomputed via
# hessian_via_grad().


wald_tests_by_category <- function(fit_full,
                                   E_min = 50, M_min = 20,
                                   tau_w = 0.1, C_min = 20,
                                   regularize = 1e-6,
                                   hessian = NULL, cov_full = NULL) {
  stopifnot(isTRUE(fit_full$full))
  K <- fit_full$K; Tnum <- fit_full$Tnum; r <- fit_full$r
  Kminus <- K - 1L; Tminus <- Tnum - 1L
  idx <- param_index(K, Tnum, r, full = TRUE)


  Y <- fit_full$Y; N <- fit_full$N; X <- fit_full$X; Z <- fit_full$Z


  # Per-cell-type eligibility (non-reference cell types only; X already
  # excludes the reference column, see compass_fit_gene()).
  if (is.null(X) || ncol(X) == 0L) {
    E_t <- numeric(0); M_t <- integer(0); per_t_ok <- logical(0)
  } else {
    E_t <- as.vector(crossprod(N, X))                                   # T-1
    M_t <- vapply(seq_len(Tminus),
                  function(t) sum((N > 0) & (X[, t] >= tau_w)),
                  integer(1))
    per_t_ok <- (E_t >= E_min) & (M_t >= M_min)
  }


  # Per-category eligibility (non-reference categories only).
  C_k <- if (Kminus > 0L) colSums(Y[, seq_len(Kminus), drop = FALSE]) else numeric(0)
  per_k_ok <- (C_k >= C_min)


  # Recover full-model parameter covariance.
  if (is.null(cov_full)) {
    if (is.null(hessian)) {
      H <- hessian_via_grad(fit_full$par, Y, N, Z, K)
    } else {
      H <- hessian
    }
    H_reg <- H + diag(regularize, nrow(H))
    V <- tryCatch(solve(H_reg), error = function(e) MASS::ginv(H_reg))
  } else {
    V <- cov_full
  }
  V <- (V + t(V)) / 2


  # Per-pair tests. We always emit one record per (t, k) so the consumer can
  # see eligibility flags; only eligible pairs receive a W_tgk / p_tgk.
  per_pair <- vector("list", Tminus * Kminus)
  pvals <- numeric(0)
  pair_keys <- character(0)
  rec_i <- 0L
  for (t in seq_len(Tminus)) {
    for (k in seq_len(Kminus)) {
      rec_i <- rec_i + 1L
      eligible <- isTRUE(per_t_ok[t]) && isTRUE(per_k_ok[k])
      blk_idx <- idx$xi_int[[k]][[t]]
      xi_hat <- fit_full$par[blk_idx]
      record <- list(t = t, k = k, eligible = eligible, df = r,
                     E_t = if (length(E_t)) E_t[t] else NA_real_,
                     M_t = if (length(M_t)) M_t[t] else NA_integer_,
                     C_k = if (length(C_k)) C_k[k] else NA_real_,
                     xi_hat = xi_hat,
                     W = NA_real_, p = NA_real_, p_holm = NA_real_,
                     V_blk = NULL)
      if (eligible) {
        V_blk <- V[blk_idx, blk_idx, drop = FALSE]
        V_blk <- (V_blk + t(V_blk)) / 2
        V_blk_reg <- V_blk + diag(regularize, nrow(V_blk))
        Wtgk <- as.numeric(crossprod(xi_hat, solve(V_blk_reg, xi_hat)))
        ptgk <- pchisq(Wtgk, df = r, lower.tail = FALSE)
        record$W <- Wtgk
        record$p <- ptgk
        record$V_blk <- V_blk
        pvals <- c(pvals, ptgk)
        pair_keys <- c(pair_keys, sprintf("%d_%d", t, k))
      }
      per_pair[[rec_i]] <- record
    }
  }


  # Holm adjustment across eligible pairs only.
  if (length(pvals) > 0L) {
    p_holm <- p.adjust(pvals, method = "holm")
    names(p_holm) <- pair_keys
    for (i in seq_along(per_pair)) {
      rec <- per_pair[[i]]
      if (rec$eligible) {
        key <- sprintf("%d_%d", rec$t, rec$k)
        per_pair[[i]]$p_holm <- as.numeric(p_holm[key])
      }
    }
  }


  summary_df <- do.call(rbind, lapply(per_pair, function(rec) {
    data.frame(t = rec$t, k = rec$k, eligible = rec$eligible,
               E_t = rec$E_t, M_t = rec$M_t, C_k = rec$C_k,
               W = rec$W, df = rec$df,
               p = rec$p, p_holm = rec$p_holm,
               stringsAsFactors = FALSE)
  }))


  list(per_pair = per_pair, summary = summary_df,
       E_t = E_t, M_t = M_t, C_k = C_k,
       per_t_ok = per_t_ok, per_k_ok = per_k_ok,
       params = list(E_min = E_min, M_min = M_min,
                     tau_w = tau_w, C_min = C_min),
       cov_full = V)
}


# ---------- Eligibility checks ----------------------------------------------


check_eligibility <- function(Y, N, W, r, K,
                              tau_w = 0.1,
                              E_min = 50, M_min = 20,
                              c1 = 10, c2 = 20, c3 = 10, c4 = 20) {
  S <- nrow(Y); Tnum <- ncol(W); Kminus <- K - 1L; Tminus <- Tnum - 1L
  n_nz <- sum(N > 0)
  C_g <- sum(N)
  d_full <- (K - 1L) * (1L + Tminus + r + Tminus * r) + 1L


  reduced_ok <- (n_nz >= c1 * Kminus) && (C_g >= c2 * Kminus)
  full_ok <- (n_nz >= c3 * d_full) && (C_g >= c4 * d_full)


  E_t <- as.vector(crossprod(N, W))
  M_t <- vapply(seq_len(Tnum),
                function(t) sum((N > 0) & (W[, t] >= tau_w)),
                integer(1))
  per_ct_ok <- (E_t >= E_min) & (M_t >= M_min)


  list(n_nz = n_nz, C_g = C_g, d_full = d_full,
       reduced_ok = reduced_ok, full_ok = full_ok,
       E_t = E_t, M_t = M_t, per_ct_ok = per_ct_ok)
}


# ---------- Numerical stability checks --------------------------------------


check_stability <- function(fit, prob_eps = 1e-4,
                            cond_thresh = 1e10,
                            rel_tol_multi = 1e-3,
                            hessian = NULL) {
  starts <- Filter(Negate(is.null), fit$starts)
  vals <- vapply(starts, function(x) x$value, numeric(1))
  multi_stable <- length(vals) >= 2 &&
    (max(vals) - min(vals)) / (abs(min(vals)) + 1) < rel_tol_multi


  pi_min <- min(fit$Pi); pi_max <- max(fit$Pi)
  prob_ok <- pi_min > prob_eps && pi_max < 1 - prob_eps


  if (is.null(hessian)) {
    H <- tryCatch(
      hessian_via_grad(fit$par, fit$Y, fit$N, fit$Z, fit$K),
      error = function(e) NULL
    )
  } else {
    H <- hessian
  }
  cond <- if (!is.null(H)) kappa(H, exact = FALSE) else NA_real_
  cond_ok <- !is.na(cond) && cond < cond_thresh


  list(multi_stable = multi_stable,
       pi_min = pi_min, pi_max = pi_max, prob_ok = prob_ok,
       cond_number = cond, cond_ok = cond_ok,
       hessian = H)
}


# ---------- End-to-end per-gene pipeline -------------------------------------


compass_fit_gene <- function(Y, N, W, B,
                             reference_ct = NULL,
                             run_full = TRUE,
                             do_eligibility = TRUE,
                             do_stability = TRUE,
                             n_starts = 3L,
                             verbose = FALSE) {
  S <- nrow(Y); K <- ncol(Y); Tnum <- ncol(W); r <- ncol(B)
  if (is.null(reference_ct)) reference_ct <- Tnum


  perm <- c(setdiff(seq_len(Tnum), reference_ct), reference_ct)
  W_p <- W[, perm, drop = FALSE]
  X <- W_p[, -Tnum, drop = FALSE]


  elig <- if (do_eligibility) {
    check_eligibility(Y, N, W_p, r, K)
  } else NULL


  fit0 <- fit_model(Y, N, X, B, K, full = FALSE,
                    n_starts = n_starts, verbose = verbose)
  st <- omnibus_score_test(Y, N, X, B, K, fit_reduced = fit0)


  fit1 <- NULL; wald <- NULL; wald_by_cat <- NULL; stab <- NULL
  if (run_full) {
    idx_full <- param_index(K, Tnum, r, full = TRUE)
    idx_red <- param_index(K, Tnum, r, full = FALSE)
    init <- rep(0, idx_full$total)
    for (k in seq_len(K - 1L)) {
      init[idx_full$alpha[k]] <- fit0$par[idx_red$alpha[k]]
      if (length(idx_full$beta[[k]]) > 0L)
        init[idx_full$beta[[k]]] <- fit0$par[idx_red$beta[[k]]]
      init[idx_full$xi0[[k]]] <- fit0$par[idx_red$xi0[[k]]]
    }
    init[idx_full$kappa] <- fit0$par[idx_red$kappa]
    fit1 <- fit_model(Y, N, X, B, K, full = TRUE,
                      n_starts = n_starts, init_params = init,
                      verbose = verbose)
    H_full <- tryCatch(
      hessian_via_grad(fit1$par, Y, N, fit1$Z, K),
      error = function(e) NULL
    )
    wald <- wald_tests(fit1, hessian = H_full)
    wald_by_cat <- tryCatch(
      wald_tests_by_category(fit1,
                             hessian = H_full,
                             cov_full = wald$cov_full),
      error = function(e) {
        if (verbose) message("wald_tests_by_category failed: ",
                             conditionMessage(e))
        NULL
      }
    )
    if (do_stability) stab <- check_stability(fit1, hessian = H_full)
  }


  list(eligibility = elig,
       fit_reduced = fit0,
       score_test = st,
       fit_full = fit1,
       wald = wald,
       wald_by_cat = wald_by_cat,
       stability = stab,
       perm = perm, reference_ct = reference_ct)
}


compass_fit_many <- function(Y_list, N_list, W, B, ..., verbose = FALSE) {
  G <- length(Y_list)
  results <- vector("list", G)
  p_omni <- numeric(G)
  for (g in seq_len(G)) {
    if (verbose) message("[compass] gene ", g, "/", G)
    results[[g]] <- tryCatch(
      compass_fit_gene(Y_list[[g]], N_list[[g]], W, B, ...),
      error = function(e) { message("gene ", g, " failed: ", conditionMessage(e)); NULL }
    )
    p_omni[g] <- if (!is.null(results[[g]])) results[[g]]$score_test$p else NA_real_
  }
  p_bh <- p.adjust(p_omni, method = "BH")
  list(results = results, p_omni = p_omni, p_omni_bh = p_bh)
}


# ---------- Self-test: verify analytic gradient -----------------------------
#
# Returns a data.frame summarising max-abs / max-rel error of the analytic
# gradient against numDeriv::grad on small synthetic problems. Useful as a
# sanity check after any change to neg_loglik / neg_loglik_grad.
compass_self_test_gradient <- function(verbose = TRUE,
                                       cases = list(
                                         list(S = 200L, K = 4L, p_per = 10L),
                                         list(S = 300L, K = 5L, p_per = 12L),
                                         list(S = 400L, K = 2L, p_per = 8L)
                                       ),
                                       n_random = 3L,
                                       seed = 20260428L,
                                       atol = 1e-5,
                                       rtol = 1e-4) {
  set.seed(seed)
  out <- list()
  ok <- TRUE
  for (ci in seq_along(cases)) {
    cs <- cases[[ci]]
    S <- cs$S; K <- cs$K; p_per <- cs$p_per
    Z <- cbind(1, matrix(rnorm(S * (p_per - 1L)), S, p_per - 1L))
    Kminus <- K - 1L
    Gamma_true <- matrix(rnorm(p_per * Kminus, sd = 0.4), p_per, Kminus)
    eta <- Z %*% Gamma_true
    eta_full <- cbind(eta, 0)
    m <- apply(eta_full, 1, max)
    e <- exp(eta_full - m); Pi <- e / rowSums(e)
    N <- rpois(S, 6) + 1L
    Y <- t(vapply(seq_len(S),
                  function(s) rmultinom(1, N[s], Pi[s, ])[, 1],
                  integer(K)))
    total <- Kminus * p_per + 1L
    for (j in seq_len(n_random)) {
      p <- rnorm(total, sd = 0.4)
      p[total] <- log(5) + rnorm(1, 0, 0.3)
      g_an <- neg_loglik_grad(p, Y, N, Z, K)
      g_nd <- numDeriv::grad(function(q) neg_loglik(q, Y, N, Z, K), p)
      diff <- g_an - g_nd
      max_abs <- max(abs(diff))
      max_rel <- max(abs(diff) / pmax(abs(g_nd), 1e-6))
      pass <- max_abs < atol || max_rel < rtol
      ok <- ok && pass
      out[[length(out) + 1L]] <- data.frame(
        case = ci, S = S, K = K, p_per = p_per,
        rep = j, max_abs = max_abs, max_rel = max_rel,
        grad_norm = sqrt(sum(g_nd^2)), pass = pass
      )
    }
  }
  res <- do.call(rbind, out)
  if (verbose) {
    cat(sprintf("compass_core v%s analytic-gradient self-test: %s\n",
                COMPASS_CORE_VERSION, if (ok) "PASS" else "FAIL"))
    print(res, row.names = FALSE)
  }
  invisible(list(results = res, ok = ok))
                                       }