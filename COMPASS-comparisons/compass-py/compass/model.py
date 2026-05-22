import jax
import jax.numpy as jnp
import jaxopt

from .likelihood import nll, softmax_ref, spot_totals
from .design import build_design
# import jax.scipy.optimize as jsp_optimize
from .indexing import ParamIndex

from .tests_classes import FitResult

from functools import lru_cache

@lru_cache()
def get_compiled_solver(maxiter,reltol,K,T,r,full):
    # idt we'll need to hash it then if static?
    idx_static = ParamIndex(K=K, T=T, r=r, full=full)

    # so Y N Z aren't baked into the compiled graph, closure over them
    def obj_fun(params, Y, N, Z):
        return nll(params, Y, N, Z, K, idx_static, include_const=False)
    solver = jaxopt.BFGS(fun=obj_fun, maxiter=maxiter, tol=reltol)

    # vmap over params (axis 0), but broadcast the data arrays (axis None
    vmap_run = jax.vmap(solver.run, in_axes=(0, None, None, None))

    return jax.jit(vmap_run)
def fit_model(
        Y:jnp.ndarray,
        N:jnp.ndarray,
        X:jnp.ndarray,
        B:jnp.ndarray,
        K:int,
        full:bool=True,
        n_starts:int=5,
        init_scale:float =0.1,
        method:str="bfgs",
        maxiter:int =500,
        reltol:float=1e-8,
        init_params:jnp.ndarray|None = None,
        verbose:bool=False,
        seed:int|None = None,
        idx:ParamIndex|None = None):

    Y = jnp.asarray(Y)
    N = spot_totals(N, Y.shape[0])

    r = B.shape[1]
    T = X.shape[1] + 1 if X is not None and X.shape[1] > 0 else 1

    if idx is not None and (idx.K, idx.T, idx.r, idx.full) != (K,T,r,full):
        raise ValueError(f"Provided idx has K={idx.K}, T={idx.T}, r={idx.r}, full={idx.full} but expected K={K}, T={T}, r={r}, full={full}")
    
    if idx is None or not isinstance(idx, ParamIndex):
        idx = ParamIndex(K=K, T=T, r=r, full=full)
    n_params = len(idx)

    Z = build_design(X,B,full)


    starts = []
    best = jnp.inf
    best_params = None
    
    method = method.upper()
    if method != "BFGS":
        raise NotImplementedError(
            "Only BFGS optimization is currently implemented. see https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.minimize.html"
        )
    
    p0s = []
    for i in range(n_starts):
        if i == 0 and init_params is not None and init_params.shape[0] == n_params:
            p0 = init_params
        elif i == 0:
            p0 = jnp.zeros(n_params)
            p0 = p0.at[idx.kappa_idx].set(jnp.log(5.0))
        else:
            key = jax.random.fold_in(jax.random.key(0 if seed is None else seed), i)
            p0 = jax.random.normal(key, shape=(n_params,)) * init_scale
            p0 = p0.at[idx.kappa_idx].set(jnp.log(5.0))
        p0s.append(p0)
    
    # (n_starts, n_params)
    batched_p0 = jnp.stack(p0s)

    compiled_solver = get_compiled_solver(maxiter, reltol, K, T, r, full)

    batched_params, batched_state = compiled_solver(batched_p0, Y, N, Z)

    losses = batched_state.value
    best_idx = jnp.argmin(losses)

    best_params = batched_params[best_idx]
    best = nll(best_params, Y, N, Z, K, idx, include_const=True)
    
    # JAXopt signifies convergence when the gradient error is below the tolerance threshold
    success = batched_state.error[best_idx] <= reltol

    if best_params is None:
        raise RuntimeError("All optimization starts failed.")

    kappa = jnp.exp(best_params[idx.kappa_idx])
    gammas = idx.gamma(best_params)
    eta = Z @ gammas
    pi = softmax_ref(eta)

    return FitResult(
        params=best_params,
        Gamma=gammas,
        kappa=kappa,
        Pi=pi,
        eta=eta,
        Z=Z,
        Y=Y,
        N=N,
        X=X,
        B=B,
        K=K,
        T=T,
        r=r,
        full=full,
        nll=jnp.array(best),
        success=success,
        paramidx=idx,
        starts=starts
    )
