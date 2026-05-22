import jax
import jax.numpy as jnp
from jax.nn import softmax
from jax.scipy.special import gammaln
from .indexing import ParamIndex

from functools import partial


def spot_totals(N: jnp.ndarray, S: int) -> jnp.ndarray:
    """Return spot totals as shape (S,), accepting a single-column array."""
    N = jnp.asarray(N)
    if N.ndim == 2 and N.shape[1] == 1:
        N = N[:, 0]
    if N.ndim != 1 or N.shape[0] != S:
        raise ValueError(f"N must have shape ({S},) or ({S}, 1); got {N.shape}")
    return N


def softmax_ref(eta: jnp.ndarray) -> jnp.ndarray:
    """
    A helper function to compute the softmax of eta with an implicit reference category.
    Args:
        eta (jnp.ndarray): (S, K-1) matrix of logits for the non-reference categories.
    Returns:
        softmax (jnp.ndarray): (S, K) matrix of probabilities, where the last column corresponds to the reference category.
    """
    # cat 0s to the isoform axis of eta
    eta_full = jnp.concatenate([eta, jnp.zeros((eta.shape[0], 1))], axis=1)
    # do a softmax over the isoform axis
    return softmax(eta_full, axis=1)



def dm_loglik(Y: jnp.ndarray,
              N: jnp.ndarray,
              pi: jnp.ndarray,
              kappa: jnp.ndarray,
              include_const:bool=True) -> jnp.ndarray:
    """
    Args:
        Y: (S,K) matrix of observed counts. 
        N: (S,) total reads per spot.
        pi: (S,K) distributions of K isoforms at each spot. rows su m to 1.
        kappa (float): gene specific concentration parameter. higher means less variable
    """

    N = spot_totals(N, Y.shape[0])

    # (S,K)
    alpha = kappa * pi
    # (S,)
    a0 = alpha.sum(axis=1)
    # (S,)
    log_likelihood = (gammaln(a0) - gammaln(N+a0)
           + (gammaln(Y+alpha)-gammaln(alpha)).sum(axis=1))
    if include_const:
        log_likelihood += gammaln(N + 1) - gammaln(Y + 1).sum(axis=1)

    # this gives warning if 64bit not on when casting to float
    return log_likelihood.sum()


@partial(jax.jit, static_argnames=["K","idx","include_const"])
def nll(
        params:jnp.ndarray,
        Y:jnp.ndarray,
        N:jnp.ndarray,
        Z:jnp.ndarray,
        K:int,
        idx:ParamIndex,
        include_const:bool=True):
    """
    Args:
        params: flattened parameters
        Y: (S, K) matrix of observed counts
        N: (S,) total reads per spot
        Z: (S, R) matrix of latent factors
        K: number of isoforms
        idx: ParamIndex object that specifies how to unpack/pack params
    Returns:
        nll: negative log likelihood
    """
    # shape: (K-1)*R -> (K-1, R)
    gammas = idx.gamma(params)
    eta = Z @ gammas
    pi = softmax_ref(eta)
    # sorta arbitrary but cant let kappa explode 
    kappa = jnp.exp(jnp.clip(params[idx.kappa_idx], a_max=20.0))
    return -dm_loglik(Y, N, pi, kappa, include_const=include_const)
