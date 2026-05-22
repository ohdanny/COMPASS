import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from compass.indexing import ParamIndex

GOLDEN = Path(__file__).parent / "param_index.json"

CASES = [
    ("case_reduced", 3, 4, 5, False),
    ("case_full",    3, 4, 5, True),
    ("case_T2",      4, 2, 3, True),
    ("case_K2",      2, 3, 4, True),
    ("case_T1",      3, 1, 5, True),
    ("case_T1_red",  3, 1, 5, False),
]


@pytest.fixture
def golden():
    
    with open(GOLDEN) as f:
        try:
            g = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to load golden data from {GOLDEN}: {e}")
    return g["R_outputs"]


def _r_to_py(arr):
    """R 1-indexed -> 0-indexed numpy, any nesting."""
    return np.asarray(arr) - 1


@pytest.mark.parametrize("case_name,K,T,r,full", CASES)
def test_param_index_matches_r(golden, case_name, K, T, r, full):
    expected = golden[case_name]
    py = ParamIndex(K=K, T=T, r=r, full=full)

    assert py.n_param_per_k == expected["p_per"]
    assert py.total_params == expected["total"]
    np.testing.assert_array_equal(np.asarray(py.alpha_idx), _r_to_py(expected["alpha"]))
    np.testing.assert_array_equal(np.asarray(py.beta_idx),  _r_to_py(expected["beta"]))
    np.testing.assert_array_equal(np.asarray(py.xi0_idx),   _r_to_py(expected["xi0"]))

    if full and T > 1:
        np.testing.assert_array_equal(
            np.asarray(py.xi_int_idx), _r_to_py(expected["xi_int"])
        )
    else:
        assert py.xi_int_idx is None

    # kappa: R gives a scalar; compare as scalar
    assert int(py.kappa_idx) == int(_r_to_py(expected["kappa"]))


"""
 2. THE layout-authority test: gamma() column k == contiguous R block k,
    and that block == [alpha | beta | xi0 | xi_int] in R order.
    This is the assertion that closes the "silent wrong science" class.
"""
@pytest.mark.parametrize("case_name,K,T,r,full", CASES)
def test_gamma_columns_are_r_contiguous_blocks(golden, case_name, K, T, r, full):
    py = ParamIndex(K=K, T=T, r=r, full=full)
    rng = np.random.default_rng(0)
    theta = jnp.asarray(rng.standard_normal(py.total_params))

    G = np.asarray(py.gamma(theta))            # (p_per, K-1)
    assert G.shape == (py.n_param_per_k, K - 1)

    p_per = py.n_param_per_k
    for k in range(K - 1):
        # column k of Gamma must be exactly theta[k*p_per:(k+1)*p_per]
        np.testing.assert_array_equal(
            G[:, k], np.asarray(theta)[k * p_per:(k + 1) * p_per]
        )


@pytest.mark.parametrize("case_name,K,T,r,full", CASES)
def test_gamma_unpack_agree(golden, case_name, K, T, r, full):
    """gamma()'s column k, sliced by sub-block, must equal unpack()'s blocks."""
    py = ParamIndex(K=K, T=T, r=r, full=full)
    rng = np.random.default_rng(1)
    theta = jnp.asarray(rng.standard_normal(py.total_params))

    G = np.asarray(py.gamma(theta))            # (p_per, K-1)
    alpha, beta, xi0, xi_int, kappa = py.unpack(theta)
    alpha, beta, xi0 = map(np.asarray, (alpha, beta, xi0))

    Tm = T - 1
    for k in range(K - 1):
        col = G[:, k]
        np.testing.assert_array_equal(col[0:1],        alpha[k:k + 1])
        np.testing.assert_array_equal(col[1:1 + Tm],   beta[k])
        np.testing.assert_array_equal(col[T:r + T],   xi0[k])  # 1+Tm == T
        if full and T > 1:
            np.testing.assert_array_equal(col[T+r:], np.asarray(xi_int)[k].reshape(-1))
        else:
            assert col.shape[0] == T + r - Tm + Tm  # sanity: no xi_int slots


# ---------------------------------------------------------------------------
# 3. unpack round-trip: flatten known blocks -> unpack -> identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_name,K,T,r,full", CASES)
def test_unpack_roundtrip(golden, case_name, K, T, r, full):
    py = ParamIndex(K=K, T=T, r=r, full=full)
    rng = np.random.default_rng(42)
    theta = jnp.asarray(rng.standard_normal(py.total_params))

    alpha, beta, xi0, xi_int, kappa = py.unpack(theta)

    assert np.asarray(alpha).shape == (K - 1,)
    assert np.asarray(beta).shape == (K - 1, T - 1)
    assert np.asarray(xi0).shape == (K - 1, r)
    if full and T > 1:
        assert xi_int is not None
        assert np.asarray(xi_int).shape == (K - 1, T - 1, r)
    else:
        assert xi_int is None

    # values at indexed positions
    np.testing.assert_array_equal(np.asarray(alpha), np.asarray(theta)[np.asarray(py.alpha_idx)])
    np.testing.assert_array_equal(np.asarray(beta),  np.asarray(theta)[np.asarray(py.beta_idx)])
    np.testing.assert_array_equal(np.asarray(xi0),   np.asarray(theta)[np.asarray(py.xi0_idx)])

    # full reconstruction: scatter blocks back, must equal theta exactly
    recon = np.empty(py.total_params)
    recon[np.asarray(py.alpha_idx)] = np.asarray(alpha)
    recon[np.asarray(py.beta_idx)]  = np.asarray(beta)
    recon[np.asarray(py.xi0_idx)]   = np.asarray(xi0)
    if xi_int is not None:
        recon[np.asarray(py.xi_int_idx)] = np.asarray(xi_int)
    recon[py.kappa_idx] = float(kappa)
    np.testing.assert_array_equal(recon, np.asarray(theta))


# ---------------------------------------------------------------------------
# 4. xi block accessors: ordering + mutual consistency (category-major)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_name,K,T,r,full",
                         [c for c in CASES if c[4] and c[2] is not None])
def test_xi_blocks_consistent(golden, case_name, K, T, r, full):
    py = ParamIndex(K=K, T=T, r=r, full=full)
    if not (full and T > 1):
        pytest.skip("no interaction block")

    # xi_block_pair(t,k) is exactly the (k,t,:) row of xi_int_idx
    for t in range(T - 1):
        for k in range(K - 1):
            np.testing.assert_array_equal(
                np.asarray(py.xi_block_pair(t, k)),
                np.asarray(py.xi_int_idx)[k, t, :],
            )

    # xi_block_celltype(t) == concat over k of xi_block_pair(t,k), category-major
    for t in range(T - 1):
        expected = np.concatenate(
            [np.asarray(py.xi_block_pair(t, k)) for k in range(K - 1)]
        )
        np.testing.assert_array_equal(
            np.asarray(py.xi_block_celltype(t)), expected
        )

    # xi_flat() == concat over t of xi_block_celltype(t)?  -> see note below
    flat = np.asarray(py.xi_flat())
    assert flat.shape == ((K - 1) * (T - 1) * r,)
    assert set(flat.tolist()) == set(np.asarray(py.xi_int_idx).reshape(-1).tolist())


# ---------------------------------------------------------------------------
# 5. reduced / T==1 / K==2 edge behavior
# ---------------------------------------------------------------------------
def test_reduced_has_no_xi_int():
    py = ParamIndex(K=3, T=4, r=5, full=False)
    assert py.xi_int_idx is None
    assert py.xi_flat() is None
    _, _, _, xi_int, _ = py.unpack(jnp.zeros(py.total_params))
    assert xi_int is None
    with pytest.raises(ValueError):
        py.xi_block_celltype(0)
    with pytest.raises(ValueError):
        py.xi_block_pair(0, 0)


def test_T1_has_no_interaction_or_beta():
    py = ParamIndex(K=3, T=1, r=5, full=True)
    assert py.xi_int_idx is None          # T>1 required
    assert np.asarray(py.beta_idx).shape == (K_minus(3), 0)  # T-1 == 0 cols


def K_minus(K):  # tiny helper for readability above
    return K - 1


# ---------------------------------------------------------------------------
# 6. eq / hash contract (needed for static jit arg)
# ---------------------------------------------------------------------------
def test_eq_hash_keyed_on_shape_only():
    a = ParamIndex(K=3, T=4, r=5, full=True)
    b = ParamIndex(K=3, T=4, r=5, full=True)
    c = ParamIndex(K=3, T=4, r=5, full=False)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert len({a, b, c}) == 2          # usable as dict/set key & jit static arg