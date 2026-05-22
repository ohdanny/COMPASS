"""Tests for compass.likelihood."""

import json
from pathlib import Path

import jax

# float64 is required to match R's neg_loglik / dm_loglik to high precision.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from compass.indexing import ParamIndex
from compass.likelihood import dm_loglik, nll, softmax_ref


GOLDEN_DIR = Path(__file__).parent


def load_golden(name: str) -> dict:
    """Load a JSON golden file; skip the test if R outputs haven't been written yet."""
    with open(GOLDEN_DIR / f"{name}.json") as f:
        g = json.load(f)
    out = g.get("R_outputs")
    if out is None or (isinstance(out, (list, dict)) and len(out) == 0):
        pytest.skip(f"Run Rscript {name}.R first to populate outputs")
    if isinstance(out, dict) and any(v is None for v in out.values()):
        pytest.skip(f"Run Rscript {name}.R first to populate outputs")
    return g


# ----- softmax_ref --------------------------------------------------------
def test_softmax_ref_rows_sum_to_one():
    eta = jnp.array([[1.0, 2.0], [0.0, -1.0]])
    pi = softmax_ref(eta)
    assert pi.shape == (eta.shape[0],eta.shape[1]+1)
    assert jnp.allclose(pi.sum(axis=1), 1.0)


def test_softmax_ref_zero_eta_is_uniform():
    eta = jnp.zeros((3, 4))  # K = 5
    pi = softmax_ref(eta)
    assert jnp.allclose(pi, 1 / 5)


def test_softmax_ref_reference_is_last():
    eta = jnp.array([[10.0, -10.0]])
    pi = softmax_ref(eta)
    assert pi[0, 0] > 0.99
    assert pi[0, -1] < 1e-3


# ----- softmax_ref vs R ---------------------------------------------------
@pytest.fixture
def softmax_golden():
    return load_golden("softmax_ref")


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_softmax_ref_matches_r(softmax_golden, idx):
    cases = softmax_golden["cases"]
    r_outs = softmax_golden["R_outputs"]
    assert len(cases) == len(r_outs), "case/output length mismatch in softmax_ref.json"

    cs, out = cases[idx], r_outs[idx]
    assert cs["name"] == out["name"], f"case order drifted at idx={idx}"

    eta = jnp.array(cs["eta"], dtype=jnp.float64)
    pi_py = np.asarray(softmax_ref(eta))
    pi_r = np.array(out["Pi"], dtype=np.float64)

    # row-stochastic sanity (Python side)
    np.testing.assert_allclose(pi_py.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        pi_py, pi_r, atol=1e-12, rtol=1e-12,
        err_msg=f"softmax_ref mismatch for case '{cs['name']}'",
    )



# ----- dm_loglik ----------------------------------------------------------
@pytest.fixture
def dm_golden():
    return load_golden("dm_loglik")


def test_dm_loglik_matches_r_with_const(dm_golden):
    inp, out = dm_golden["inputs"], dm_golden["R_outputs"]
    Y = jnp.array(inp["Y"], dtype=jnp.float64)
    pi = jnp.array(inp["Pi"], dtype=jnp.float64)
    N = Y.sum(axis=1)
    val = dm_loglik(Y, N, pi, inp["kappa"], include_const=True)
    assert abs(val - out["with_const"]) < 1e-10


def test_dm_loglik_matches_r_without_const(dm_golden):
    inp, out = dm_golden["inputs"], dm_golden["R_outputs"]
    Y = jnp.array(inp["Y"], dtype=jnp.float64)
    pi = jnp.array(inp["Pi"], dtype=jnp.float64)
    N = Y.sum(axis=1)
    val = dm_loglik(Y, N, pi, inp["kappa"], include_const=False)
    assert abs(val - out["without_const"]) < 1e-10


def test_dm_loglik_const_diff_is_multinomial_coef(dm_golden):
    """include_const=True vs =False should differ by the multinomial coef."""
    from jax.scipy.special import gammaln

    inp = dm_golden["inputs"]
    Y = jnp.array(inp["Y"], dtype=jnp.float64)
    pi = jnp.array(inp["Pi"], dtype=jnp.float64)
    N = Y.sum(axis=1)
    diff = (dm_loglik(Y, N, pi, inp["kappa"], True)
           - dm_loglik(Y, N, pi, inp["kappa"], False))
    expected = (gammaln(N + 1) - gammaln(Y + 1).sum(axis=1)).sum()
    assert abs(diff - expected) < 1e-10


def test_dm_loglik_accepts_single_column_N(dm_golden):
    """A column-vector N should not broadcast against a0 into an S x S array."""
    inp = dm_golden["inputs"]
    Y = jnp.array(inp["Y"], dtype=jnp.float64)
    pi = jnp.array(inp["Pi"], dtype=jnp.float64)
    N = Y.sum(axis=1)

    val_vec = dm_loglik(Y, N, pi, inp["kappa"], include_const=True)
    val_col = dm_loglik(Y, N[:, None], pi, inp["kappa"], include_const=True)

    assert jnp.allclose(val_col, val_vec, atol=1e-12, rtol=1e-12)


# ----- nll vs R neg_loglik ------------------------------------------------
# Python `nll` (likelihood.py) and R `neg_loglik` (compass_core.R) must agree
# on identical (params, Y, N, Z, K) inputs whenever eta = Z @ Gamma stays inside
# R's [-40, 40] clip range (Python has no clip; the R-side generator in nll.R
# asserts the un-clipped regime so this comparison is well-defined).
@pytest.fixture
def nll_golden():
    return load_golden("nll")


def _nll_case_ids(golden):
    return [c["name"] for c in golden["R_outputs"]]


@pytest.mark.parametrize("case_idx", range(4))
def test_nll_matches_r(nll_golden, case_idx):
    cases = nll_golden["R_outputs"]
    if case_idx >= len(cases):
        pytest.skip(f"nll.json only has {len(cases)} cases")
    case = cases[case_idx]

    K = int(case["K"])
    T = int(case["T"])
    r = int(case["r"])
    full = bool(case["full"])

    idx = ParamIndex(K=K, T=T, r=r, full=full)

    params = jnp.array(case["params"], dtype=jnp.float64)
    Y = jnp.array(case["Y"], dtype=jnp.float64)
    N = jnp.array(case["N"], dtype=jnp.float64)
    Z = jnp.array(case["Z"], dtype=jnp.float64)

    # Shape sanity against the Python index.
    assert params.shape == (idx.total_params,), (
        f"case {case['name']}: params length {params.shape[0]} != "
        f"ParamIndex.total_params {idx.total_params}"
    )
    assert Z.shape[1] == idx.n_param_per_k, (
        f"case {case['name']}: Z has {Z.shape[1]} cols but ParamIndex "
        f"expects p_per = {idx.n_param_per_k}"
    )

    val_py = float(nll(params, Y, N, Z, K, idx))
    val_r = float(case["R_output"])

    assert abs(val_py - val_r) < 1e-8, (
        f"case {case['name']}: py={val_py!r}, R={val_r!r}, "
        f"diff={val_py - val_r:.3e}"
    )


def test_nll_changes_with_params(nll_golden):
    """Smoke test: perturbing params should change nll (not silently no-op)."""
    case = nll_golden["R_outputs"][0]
    K = int(case["K"]); T = int(case["T"]); r = int(case["r"])
    idx = ParamIndex(K=K, T=T, r=r, full=bool(case["full"]))

    params = jnp.array(case["params"], dtype=jnp.float64)
    Y = jnp.array(case["Y"], dtype=jnp.float64)
    N = jnp.array(case["N"], dtype=jnp.float64)
    Z = jnp.array(case["Z"], dtype=jnp.float64)

    base = float(nll(params, Y, N, Z, K, idx))
    perturbed = params.at[0].add(0.5)
    bumped = float(nll(perturbed, Y, N, Z, K, idx))
    assert abs(bumped - base) > 1e-6
