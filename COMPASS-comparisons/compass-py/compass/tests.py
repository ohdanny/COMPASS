import jax.numpy as jnp
import jax
import statsmodels.stats.multitest as smm

from .indexing import ParamIndex
from .model import fit_model
from .likelihood import nll,sample_Y
from .design import build_design

import numpy as np

from .tests_classes import (
    OmnibusResult,
    WaldOneResult,
    WaldResults,
    WaldOneByCategoryResult,
    WaldByCategoryResults,
    FitResult
    )


def _score_Q(
    fit_reduced:FitResult,
    Y:jnp.ndarray,
    N:jnp.ndarray,
    Z:jnp.ndarray,
    K:int,
    idx_full:ParamIndex,
    idx_reduced:ParamIndex,
    regularize:float=1e-6
):
    params_full = jnp.zeros(idx_full.total_params)
    params_full = params_full.at[idx_full.alpha_idx].set(fit_reduced.params[idx_reduced.alpha_idx])
    params_full = params_full.at[idx_full.beta_idx].set(fit_reduced.params[idx_reduced.beta_idx])
    params_full = params_full.at[idx_full.xi0_idx].set(fit_reduced.params[idx_reduced.xi0_idx])
    params_full = params_full.at[idx_full.kappa_idx].set(fit_reduced.params[idx_reduced.kappa_idx])

    grad = jax.grad(
        lambda params: nll(params, Y, N, Z, K, idx_full, include_const=False)
    )(params_full)
    hess = jax.hessian(
        lambda params: nll(params, Y, N, Z, K, idx_full, include_const=False)
    )(params_full)

    xi_idx = idx_full.xi_flat()
    if xi_idx is None:
        raise ValueError("xi_idx should not be None for full model")
    all_idx = jnp.arange(idx_full.total_params)
    theta_idx = jnp.setdiff1d(all_idx, xi_idx)

    g_xi = grad[xi_idx]
    g_theta = grad[theta_idx]

    
    I_XX = hess[xi_idx][:, xi_idx]
    I_Xt = hess[xi_idx][:, theta_idx]
    I_tt = hess[theta_idx][:, theta_idx]
    I_tt_reg = I_tt + regularize * jnp.eye(I_tt.shape[0])

    U = -g_xi
    J = I_XX - I_Xt @ jnp.linalg.solve(I_tt_reg, I_Xt.T)
    J = (J + J.T) / 2 # make sure J is symmetric
    
    eigvals, eigvecs = jnp.linalg.eigh(J)
    
    floor = jnp.maximum(1e-8, jnp.max(eigvals) * 1e-6)   # tighter relative floor
    eigvals = jnp.clip(eigvals, a_min=floor)

    J_inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T

    Q = U.T @ J_inv @ U
    return Q, eigvals, U, J, I_XX, I_Xt, I_tt, xi_idx

def omnibus_score_test(
    Y:jnp.ndarray,
    N:jnp.ndarray,
    X:jnp.ndarray,
    B:jnp.ndarray,
    K:int,
    fit_reduced :FitResult|None = None,
    n_starts:int=3,
    regularize:float=1e-6,
    bartlett:bool = False,
    bartlett_n_bootstrap:int = 200,
    bartlett_seed: int | None = None,
    bartlett_n_starts:int = 1,
    bartlett_warm_start: bool = True
):
    if fit_reduced is None:
        fit_reduced = fit_model(
            Y=Y,
            N=N,
            X=X,
            B=B,
            K=K,
            full=False,
            n_starts=n_starts,
        )

    Z = build_design(X, B, full=True)
    idx_full = ParamIndex(K=K, T=X.shape[1]+1, r=B.shape[1], full=True)
    idx_reduced = ParamIndex(K=K, T=X.shape[1]+1, r=B.shape[1], full=False)

    Q, eig, U, J, I_XX, I_Xt, I_tt, xi_idx = _score_Q(
        fit_reduced=fit_reduced,
        Y=Y,
        N=N,
        Z=Z,
        K=K,
        idx_full=idx_full,
        idx_reduced=idx_reduced,
        regularize=regularize
    )
    df = len(xi_idx)
    if not bartlett:
        pval = jax.scipy.stats.chi2.sf(Q, df=df)
    else:
        Pi = fit_reduced.Pi
        kappa = fit_reduced.kappa
        init_warm = fit_reduced.params if bartlett_warm_start else None

        Q_bootstrap = []
        rng = np.random.default_rng(bartlett_seed)
        c_hat = 1.0
        for _ in range(bartlett_n_bootstrap):
            Y_boot = sample_Y(Pi=Pi, N=N, kappa=kappa, rng=rng)
            fit_b = fit_model(Y=Y_boot, N=N, X=X, B=B, K=K, full=False, n_starts=bartlett_n_starts, init_params=init_warm)
            Q_boot = _score_Q(fit_reduced=fit_b, Y=Y_boot, N=N, Z=Z, K=K, idx_full=idx_full, idx_reduced=idx_reduced, regularize=regularize)[0]
            Q_bootstrap.append(Q_boot)


        Q_bootstrap = jnp.array(Q_bootstrap)
        
        Q_bootstrap = Q_bootstrap[jnp.isfinite(Q_bootstrap)]
        if len(Q_bootstrap)>=max(10, bartlett_n_bootstrap//5):
            c_hat = jnp.mean(Q_bootstrap) / df

        pval = jax.scipy.stats.chi2.sf(Q / c_hat, df=df)

    return OmnibusResult(
        Q=Q,
        df=len(xi_idx),
        pval=float(pval),
        U=U,
        J=J,
        I_XX=I_XX,
        I_Xt=I_Xt,
        I_tt=I_tt,
        eigvals=eig,
        fit_reduced=fit_reduced,
    )



def wald_tests(
        fit_full:FitResult,
        hessian:jnp.ndarray|None = None,
        regularize:float=1e-6
):
    if not fit_full.full:
        raise ValueError("fit_full must be a full model fit")
    K,T = fit_full.K, fit_full.T
    idx = fit_full.paramidx
    Y,N,Z = fit_full.Y, fit_full.N, fit_full.Z

    if hessian is None:
        hessian = jax.hessian(
            lambda params: nll(params, Y, N, Z, K, idx, include_const=False)
        )(fit_full.params)
    assert hessian is not None, "hessian should not be None after computation" # for type checker crying

    hessian_reg = hessian + regularize * jnp.eye(hessian.shape[0])
    V = jnp.linalg.inv(hessian_reg)
    # V = (V + V.T) / 2 idt i need this 
    
    outs: list[WaldOneResult] = []

    for t in range(T-1):
        blk_idx = idx.xi_block_celltype(t)
        Xi_hat = fit_full.params[blk_idx]
        V_blk = V[blk_idx][:, blk_idx]
        V_blk = (V_blk + V_blk.T) / 2 # ensure symmetry
        V_blk = V_blk + regularize * jnp.eye(V_blk.shape[0])
        W = Xi_hat.T @ jnp.linalg.solve(V_blk, Xi_hat)
        df_t = len(blk_idx)
        p_t = 1 - jax.scipy.stats.chi2.cdf(W, df=df_t)
        out = WaldOneResult(
            t=t,
            pval=float(p_t),
            df=df_t,
            W=W,
            Xi_hat=Xi_hat,
            V_blk=V_blk
        )
        outs.append(out)
    ps = jnp.array([rec.pval for rec in outs])
    p_holm = smm.multipletests(ps, method="holm")[1]
    [setattr(out, 'p_holm', float(p_holm[i])) for i, out in enumerate(outs)]
    return WaldResults(
        results=outs,
        I_full=hessian,
        cov_full=V
    )

def wald_tests_by_category(
        fit_full:FitResult,
        E_min:float=50,
        M_min:int=20,
        tau_w:float=0.1,
        C_min:int=20,
        regularize:float=1e-6,
        hessian=None,
        cov_full=None
):
    if not fit_full.full:
        raise ValueError("fit_full must be a full model fit")
    K,T,r = fit_full.K, fit_full.T, fit_full.r
    idx = fit_full.paramidx
    Y,N,Z,X= fit_full.Y, fit_full.N, fit_full.Z, fit_full.X

    E_t = (N @ X).reshape(-1)
    M_t = jnp.sum((N[:, None] > 0) & (X >= tau_w), axis=0)  # shape (T-1,)
    per_t_ok = (E_t >= E_min) & (M_t >= M_min)

    C_k = Y[:,:(K-1)].sum(axis=0)
    per_k_ok = C_k >= C_min

    if hessian is None and cov_full is None:
            hessian = jax.hessian(
                lambda params: nll(params, Y, N, Z, K, idx, include_const=False)
            )(fit_full.params)
            cov_full = jnp.linalg.inv(hessian + regularize * jnp.eye(hessian.shape[0]))
    elif hessian is None or cov_full is None:
        raise ValueError("Both hessian and cov_full must be provided together")

    cov_full = (cov_full + cov_full.T) / 2 # make sure cov_full is symmetric

    per_pair = []
    pvals = []
    pair_keys = []
    for t in range(T-1):
        for k in range(K-1):
            eligible = (per_t_ok[t] and per_k_ok[k]).item()

            blk_idx = idx.xi_block_pair(t, k)
            xi_hat = fit_full.params[blk_idx]
            out = WaldOneByCategoryResult(
                t=t,
                k=k,
                eligible=eligible,
                df=len(blk_idx),
                E_t=int(E_t[t]),
                M_t=int(M_t[t]),
                C_k=int(C_k[k]),
                xi_hat=xi_hat,
                W=None,  #type: ignore
                pval=None,  #type: ignore
                p_holm=None,  #type: ignore
                V_blk=cov_full[blk_idx][:, blk_idx],
            )
            if not eligible:
                per_pair.append(out)
                continue
            V_blk = cov_full[blk_idx][:, blk_idx]
            V_blk = (V_blk + V_blk.T) / 2 
            V_blk_reg = V_blk + regularize * jnp.eye(V_blk.shape[0])
            Wtgk = xi_hat.T @ jnp.linalg.solve(V_blk_reg, xi_hat)
            p_tgk = 1 - jax.scipy.stats.chi2.cdf(Wtgk, df=r)
            out.W = Wtgk
            out.pval = float(p_tgk)
            out.V_blk = V_blk
            pvals.append(p_tgk)
            pair_keys.append((t,k))

            per_pair.append(out)

    if len(pvals) > 0:
        pvals = jnp.asarray(pvals)
        p_holm = smm.multipletests(jnp.asarray(pvals), method="holm")[1] 
        holm_by_pair = {tk: float(ph) for tk, ph in zip(pair_keys, p_holm)}
        for rec in per_pair:
            if rec.eligible:
                rec.p_holm = holm_by_pair[(rec.t, rec.k)]

    return WaldByCategoryResults(
        per_pair=per_pair,
        E_t=float(E_t.mean()),
        M_t=float(M_t.mean()),
        C_k=float(C_k.mean()),
        per_t_ok=per_t_ok.tolist(),
        per_k_ok=per_k_ok.tolist(),
        params={"E_min": E_min, "M_min": M_min, "tau_w": tau_w, "C_min": C_min},
        cov_full=cov_full
    )

