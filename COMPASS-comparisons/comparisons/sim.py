import numpy as np
from config import (
    S,K,ACTIVE_INTERACTIONS
)

import jax.numpy as jnp
from compass.likelihood import softmax_ref
def rand_unit(rng, d):
    x = rng.normal(size=(d,))
    return x / np.linalg.norm(x)


def build_grid(grid_n):
    u, v = np.meshgrid(
        np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n), indexing="ij"
    )
    return np.asarray(np.column_stack([u.ravel(), v.ravel()]))

def build_regions(coords, n_regions=3):
    """Vertical stripes by x-coordinate. Returns (S, n_regions - 1) dummy-coded R.
    Last region (rightmost stripe) is the reference."""
    edges = np.linspace(0, 1, n_regions + 1)[1:-1]
    labels = np.digitize(coords[:, 0], edges)  # values in {0, ..., n_regions - 1}
    R = np.zeros((coords.shape[0], n_regions - 1))
    for j in range(n_regions - 1):
        R[:, j] = (labels == j).astype(float)
    return R


def build_W(coords, rng):
    """creates the cell type distribution matrix """
    def gauss_bump(coords, ctr, sig):
        d2 = (coords[:, 0] - ctr[0]) ** 2 + (coords[:, 1] - ctr[1]) ** 2
        return np.exp(-d2 / (2 * sig**2))

    ct1_dist = 0.8 + 2.5 * gauss_bump(coords, (0.25, 0.25), 0.22)
    ct2_dist = 0.8 + 2.5 * gauss_bump(coords, (0.75, 0.75), 0.22)
    W_raw = np.column_stack([ct1_dist, ct2_dist, np.full(S, 1.0)])
    W_smooth = W_raw / W_raw.sum(1, keepdims=True)
    # adds dirichlet jitter so W isn't perfectly smooth
    W = np.vstack([rng.dirichlet(30.0 * p + 0.01) for p in W_smooth])
    # (S,T)
    return jnp.asarray(W)


def sample_N(
    a_g: float,
    R: np.ndarray,
    B: np.ndarray,
    rng: np.random.Generator,
    u_g: np.ndarray | None = None,
    gamma_scale: float | None = 0.5,
    gamma: np.ndarray | None = None,
    spatial: bool = False,
):
    """
    samples N from a spatial or nonspatial model
    """
    S = R.shape[0]

    # arbitrary scale might need to be fixed
    eps = rng.normal(scale=0.2, size=S)

    if not spatial:
        log_lam = a_g + eps

    else:    
        # spatial: log λ_gs = a_g + B·γ + ε
        if gamma is None:
            if gamma_scale is None:
                raise ValueError("Must provide either gamma or gamma_scale when spatial=True")
            #random unit vector scaled by gamma_scale
            gamma = rand_unit(rng, R.shape[1]) * gamma_scale

        if u_g is None:
            # wtf is u_g supposed to be? later problem to setup ICAR and proper region
            # stuff with R 
            u_g = 0.5 * B @ rand_unit(rng, B.shape[1])   

        # IMPORTANT: forgetting ab R for now bc idk what to do for regions
        log_lam = a_g + u_g + eps + R@gamma
    lam = np.exp(log_lam)
    return np.maximum(5, rng.poisson(lam).astype(np.int64))

def build_truth(
    expr,
    iso,
    X,
    B,
    r,
    true_params,
    rep,
    rng):
    alpha_true, beta_true, xi0_true, xi_int_true, kappa_true = [
        true_params[k] for k in ["alpha", "beta", "xi0", "xi_int", "kappa"]
    ]
    S,r = B.shape


    def active_set(iso_setting):
        return ACTIVE_INTERACTIONS.get(iso_setting, set())

    xi_int = np.zeros((r, X.shape[1], K - 1))

    if iso == "strict_null":
        alpha = alpha_true
        beta = np.zeros((K - 1, X.shape[1]))
        xi0 = np.zeros((r, K - 1))
    elif iso == "shared_null":
        alpha = alpha_true
        beta = beta_true
        xi0 = xi0_true
        xi_int = xi_int
    elif iso == "single_interaction" or iso == "two_interaction":
        alpha = alpha_true
        beta = beta_true
        xi0 = xi0_true
        active = active_set(iso)
        for t in range(X.shape[1]):
            for k in range(K - 1):
                if (t, k) in active:
                    xi_int[:, t, k] = xi_int_true[:, t, k]
    else:
        raise ValueError(iso)
    

    return dict(
        alpha=alpha, beta=beta, xi0=xi0, xi_int=xi_int, kappa=kappa_true #type: ignore
    )


def simulate_from_truth(truth, X, B, N, rng):
    S = X.shape[0]
    
    alpha, beta, xi0, xi_int, kappa = [
        truth[k] for k in ["alpha", "beta", "xi0", "xi_int", "kappa"]
    ]

    eta = np.zeros((S, K - 1))
    for k in range(K - 1):
        eta[:, k] += alpha[k] + X @ beta[k] + B @ xi0[:, k]
        for t in range(X.shape[1]):
            eta[:, k] += X[:, t] * (B @ xi_int[:, t, k])
    # (S,K)
    Pi = softmax_ref(jnp.asarray(eta))
    Y = np.zeros((S, Pi.shape[1]), np.int64)
    for s in range(S):
        if N[s] == 0:
            continue
        g = rng.gamma(np.maximum(kappa * Pi[s], 1e-8), 1.0)
        Y[s] = rng.multinomial(N[s], g / g.sum())
    return jnp.asarray(Y), Pi


def simulate_counts(a_g,R,X,B,expr_setting,iso_setting,rep_id,rng ,params=None):

    N = sample_N(a_g=a_g,R=R,B=B,rng=rng,spatial=True if expr_setting=="spatial" else False)

    true_params = params if params is not None else load_true_params(T=X.shape[1]+1, r=B.shape[1], K=K)

    truth = build_truth(
        expr=expr_setting,
        iso=iso_setting,
        X=X,
        B=B,
        r=B.shape[1],
        true_params=true_params, #type: ignore
        rep=rep_id,
        rng=rng,
    )
    Y, Pi = simulate_from_truth(truth, X, B, N, rng)
    return Y, Pi, N


def load_true_params(T, r, K):
    
    g0 = np.random.default_rng(67)

    true_params = dict(
        alpha=np.array([0.3, -0.2]),
        beta=np.array([[0.8, -0.5], [-0.6, 0.4]]),
        xi0=np.column_stack([0.5 * rand_unit(g0, r) for _ in range(K - 1)]),
        xi_int=np.stack(
            [
                np.column_stack([1.5 * rand_unit(g0, r) for _ in range(K - 1)])
                for _ in range(T-1)
            ],
            axis=1,
        ),  # shape (r, T-1, K-1)
        kappa=25.0,
    )
    return true_params
