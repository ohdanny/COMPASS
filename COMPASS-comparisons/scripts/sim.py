import numpy as np
import jax.numpy as jnp

from COMPASS.compass.likelihood import softmax_ref
from config import K, S,N_POIS,N_BASE,SCENARIOS

def build_grid(grid_n):
    u, v = np.meshgrid(
        np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n), indexing="ij"
    )
    return jnp.asarray(np.column_stack([u.ravel(), v.ravel()]))


def build_W(coords, rng):
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
    return W


def build_truth(
    scenario,
    X_default,
    B,
    r,
    rep,
    rng,
    kappa_true=25,
    alpha_true=[0.3, -0.2],
    beta_true=[[0.8, -0.5], [-0.6, 0.4]],  # (K-1) x (T-1)
    xi0_scale=0.5,
    xi_int_scale=1.5,
):
    g0 = np.random.default_rng(20000 + rep)
    S, r = B.shape

    def rand_unit(rng, d):
        x = rng.normal(size=(d,))
        return x / np.linalg.norm(x)

    xi0 = np.column_stack([xi0_scale * rand_unit(g0, r) for _ in range(K - 1)])
    xi_int = np.zeros((r, X_default.shape[1], K - 1))
    if scenario == "single_interaction":
        xi_int[:, 0, 0] = xi_int_scale * rand_unit(rng, r)
    elif scenario == "two_interaction":
        xi_int[:, 0, 0] = xi_int_scale * rand_unit(rng, r)
        xi_int[:, 1, 1] = xi_int_scale * rand_unit(rng, r)
    elif scenario != "null":
        raise ValueError(scenario)
    return dict(
        alpha=alpha_true, beta=beta_true, xi0=xi0, xi_int=xi_int, kappa=kappa_true
    )


def simulate_counts(truth, X_default, B, N, rng):
    alpha, beta, xi0, xi_int, kappa = [
        truth[k] for k in ["alpha", "beta", "xi0", "xi_int", "kappa"]
    ]

    eta = np.zeros((S, K - 1))
    for k in range(K - 1):
        eta[:, k] += alpha[k] + X_default @ beta[k] + B @ xi0[:, k]
        for t in range(X_default.shape[1]):
            eta[:, k] += X_default[:, t] * (B @ xi_int[:, t, k])
    # (S,K)
    Pi = softmax_ref(jnp.asarray(eta))
    Y = np.zeros((S, Pi.shape[1]), np.int64)
    for s in range(S):
        if N[s] == 0:
            continue
        g = rng.gamma(np.maximum(kappa * Pi[s], 1e-8), 1.0)
        Y[s] = rng.multinomial(N[s], g / g.sum())
    return Y, Pi




def compute_power(rows, alpha=0.05):
    out = []
    for sc in ("single_interaction", "two_interaction"):
        active = {"t1": True, "t2": sc == "two_interaction"}
        for ct, pcol in (("t1", "p_t1"), ("t2", "p_t2")):
            pv = np.array(
                [r[pcol] for r in rows if r["scenario"] == sc and np.isfinite(r[pcol])]
            )
            out.append(
                dict(
                    scenario=sc,
                    cell_type=ct,
                    active_truth=active[ct],
                    n_rep=len(pv),
                    power=float(np.mean(pv < alpha)) if len(pv) else float("nan"),
                )
            )
    return out




def eta_components(truth, X, B):
    """Return dict of (S, K-1) arrays: mix, sh, int1, int2, eta. eta == sum of others."""
    S = B.shape[0]
    Km = X.shape[1] + 1 - 1  # = K-1; or pass K
    alpha, beta, xi0, xi_int = (
        truth["alpha"],
        truth["beta"],
        truth["xi0"],
        truth["xi_int"],
    )
    Km = len(alpha)
    mix = np.zeros((S, Km))
    sh = np.zeros((S, Km))
    int1 = np.zeros((S, Km))
    int2 = np.zeros((S, Km))
    for k in range(Km):
        mix[:, k] = alpha[k] + X @ np.asarray(beta)[k]
        sh[:, k] = B @ xi0[:, k]
        int1[:, k] = X[:, 0] * (B @ xi_int[:, 0, k])
        int2[:, k] = X[:, 1] * (B @ xi_int[:, 1, k])
    eta = mix + sh + int1 + int2
    return dict(mix=mix, sh=sh, int1=int1, int2=int2, eta=eta)


def setup_pi_y_plots(
        X_default, B_default, r_default, rng
):
    all_pi, pi_titles, all_props, prop_titles = [], [], [], []
    eta_by_scenario = {}
    N_sim = np.maximum(5, N_BASE + rng.poisson(N_POIS, size=S)).astype(np.int64)
    for sc in SCENARIOS:
        truth = build_truth(
            scenario=sc, X_default=X_default, B=B_default, r=r_default, rep=1, rng=rng
        )
        Y, Pi = simulate_counts(truth, X_default, B_default, N_sim, rng)
        eta_by_scenario[sc] = eta_components(
            truth, X_default, B_default
        )  # <-- same truth
        all_pi.append(Pi)
        props = Y / N_sim[:, None]
        all_props.append(props)
        for k in range(K):
            iso = f"isoform k={k+1}" if k < K - 1 else f"isoform k={k+1} (ref)"
            pi_titles.append(f"{sc}\n{iso}")
            prop_titles.append(f"{sc}\n{iso}")
    combined_pi = np.hstack(all_pi)
    combined_props = np.hstack(all_props)
    return combined_pi, pi_titles, combined_props, prop_titles, eta_by_scenario
