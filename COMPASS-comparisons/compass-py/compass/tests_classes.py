from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
import pandas as pd
from jax.scipy.optimize import OptimizeResults

from .indexing import ParamIndex


@dataclass
class OmnibusResult:
    Q: jnp.ndarray
    df: int
    pval: float
    U: jnp.ndarray
    J: jnp.ndarray
    I_XX: jnp.ndarray
    I_Xt: jnp.ndarray
    I_tt: jnp.ndarray
    eigvals: jnp.ndarray
    fit_reduced: Any

@dataclass
class WaldOneResult:
    t: int
    pval: float
    df: int
    W: jnp.ndarray
    Xi_hat: jnp.ndarray
    V_blk: jnp.ndarray
    pval_holm: float | None = None
    
@dataclass
class WaldResults:
    results: list[WaldOneResult]
    I_full: jnp.ndarray
    cov_full: jnp.ndarray


@dataclass
class WaldOneByCategoryResult:
    t:int
    k:int
    eligible:bool
    df:int
    E_t:float
    M_t:int
    C_k:float
    xi_hat: jnp.ndarray
    W: jnp.ndarray
    pval: float
    p_holm: float
    V_blk: jnp.ndarray


@dataclass
class WaldByCategoryResults:
    per_pair: list[WaldOneByCategoryResult]
    E_t: float
    M_t: float
    C_k: float
    per_t_ok: list[bool]
    per_k_ok: list[bool]
    params:dict[str, Any]
    cov_full: jnp.ndarray

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"t": r.t, "k": r.k, "eligible": r.eligible,
             "E_t": r.E_t, "M_t": r.M_t, "C_k": r.C_k,
             "W": _f(r.W), "df": r.df,
             "p": _f(r.pval), "p_holm": _f(r.p_holm)}
            for r in self.per_pair
        ])


def _f(x):
    # NA-safe scalar coercion: ineligible pairs carry None/NaN through unchanged
    if x is None:
        return jnp.nan
    return float(x)


@dataclass
class FitResult:
    params: jnp.ndarray
    Gamma: jnp.ndarray
    kappa: jnp.ndarray
    Pi: jnp.ndarray
    eta: jnp.ndarray
    Z: jnp.ndarray
    Y: jnp.ndarray
    N: jnp.ndarray
    X: jnp.ndarray
    B: jnp.ndarray
    K: int
    T: int
    r: int
    full: bool
    nll: jnp.ndarray
    success: bool
    paramidx: ParamIndex
    starts: list[OptimizeResults]

    def __post_init__(self):
        self.log_likelihood = -self.nll
    

@dataclass
class CompassResult:
    fit_reduced: FitResult
    fit_full: FitResult | None
    score_test: OmnibusResult
    wald: WaldResults | None
    wald_by_cat: WaldByCategoryResults | None

    perm: jnp.ndarray | None
    reference_ct: int | None

    stability: Any = None
    eligibility: Any = None

@dataclass
class CompassManyResult:
    results: list[CompassResult]
    p_omnibus: list[float]
    p_omnibus_bh: list[float]