import jax.numpy as jnp


def build_design(
        X: jnp.ndarray,
        B: jnp.ndarray,
        full:bool=True):
    """
    Args:
        X (jnp.ndarray): (S,T-1) cell type matrix
        B (jnp.ndarray): (S,r) spatial basis matrix
        full (bool): whether to include interaction of cell type and spatial basis
    Returns:
        Z (jnp.ndarray): (S, R) design matrix, where R depends on whether full is True or False.
    """
    S = B.shape[0]
    # r = B.shape[1]
    if X is None or X.shape[1] == 0:
        B = jnp.column_stack([jnp.ones(S), B])
        return B
    
    Z = jnp.column_stack([jnp.ones(S), X, B])
    if not full:
        return Z
    
    # (S,T-1) * (S,r) -> (S,T-1,r) -> (S,(T-1)*r)
    interactions = (X[:, :, None] * B[:, None, :]).reshape(S, -1)
    return jnp.column_stack([Z, interactions])
