import jax.numpy as jnp
import jax
import statsmodels.stats.multitest as smm

from .model import fit_model
from .tests import omnibus_score_test,wald_tests,wald_tests_by_category
from .indexing import ParamIndex
from .tests_classes import CompassResult,CompassManyResult
from .likelihood import nll

def compass_fit_gene(
        Y:jnp.ndarray,
        N:jnp.ndarray,
        W:jnp.ndarray,
        B:jnp.ndarray,
        reference_ct:int|None = None,
        run_full:bool=True,
        n_starts:int=3,
        verbose:bool=False,
):

    S,K = Y.shape
    T = W.shape[1]
    r = B.shape[1]
    
    ref_ct = reference_ct if reference_ct is not None else T - 1  # 0-based!
    other_ct = [t for t in range(T) if t != ref_ct]
    # perm = jnp.array([ref_ct] + other_ct)
    perm = jnp.array(other_ct + [ref_ct])
    W_p = W[:, perm]
    X = W_p[:,:-1]

    idx_reduced = ParamIndex(K=K, T=T, r=r, full=False)
    fit0 = fit_model(
        Y=Y,
        N=N,
        X=X,
        B=B,
        K=K,
        full=False,
        n_starts=n_starts,
        verbose=verbose
    )
    st = omnibus_score_test(
        Y=Y,
        N=N,
        X=X,
        B=B,
        K=K,
        fit_reduced=fit0,
        n_starts=n_starts,
    )
    fit1,wald,wald_by_cat,stab = None, None, None, None
    if run_full:
        idx_full = ParamIndex(K=K, T=T, r=r, full=True)
        init_params = jnp.zeros(idx_full.total_params)
        init_params = init_params.at[idx_full.alpha_idx].set(fit0.params[idx_reduced.alpha_idx])
        init_params = init_params.at[idx_full.beta_idx].set(fit0.params[idx_reduced.beta_idx])
        init_params = init_params.at[idx_full.xi0_idx].set(fit0.params[idx_reduced.xi0_idx])
        init_params = init_params.at[idx_full.kappa_idx].set(fit0.params[idx_reduced.kappa_idx])
        fit1 = fit_model(
            Y=Y,
            N=N,
            X=X,
            B=B,
            K=K,
            full=True,
            n_starts=n_starts,
            init_params=init_params,
            verbose=verbose
        )
        hessian_full = jax.hessian(
            lambda params: nll(
                params, fit1.Y, fit1.N, fit1.Z, fit1.K, fit1.paramidx,
                include_const=False,
            )
        )(fit1.params)
        wald = wald_tests(fit1, hessian_full)
        wald_by_cat = wald_tests_by_category(fit1, hessian=hessian_full,
                                             cov_full=wald.cov_full)

    return CompassResult(
        fit_reduced=fit0,
        fit_full=fit1,
        score_test=st,
        wald=wald,
        wald_by_cat=wald_by_cat,
        perm=perm,
        reference_ct=ref_ct,
        stability=stab
    )


def compass_fit_many(
        Y:jnp.ndarray,
        N:jnp.ndarray,
        W:jnp.ndarray,
        B:jnp.ndarray,
        reference_ct:int|None = None,
        run_full:bool=True,
        n_starts:int=3,
        verbose:bool=False,
):
    G = Y.shape[1]
    results = []
    p_omni = []
    for g in range(G):
        if verbose:
            print(f"Fitting gene {g+1}/{G}")
        res = compass_fit_gene(
            Y=Y[:,g:g+1],
            N=N[:,g:g+1],
            W=W,
            B=B,
            reference_ct=reference_ct,
            run_full=run_full,
            n_starts=n_starts,
            verbose=verbose
        )
        results.append(res)
        p_omni.append(res.score_test.pval)  
    p_omni = jnp.array(p_omni)
    p_bh = smm.multipletests(p_omni, method='fdr_bh')[1]

    return CompassManyResult(
        results=results,
        p_omnibus=p_omni.tolist(),
        p_omnibus_bh=p_bh.tolist()
    )
