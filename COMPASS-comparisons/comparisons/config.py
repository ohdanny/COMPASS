# config.py  -- constants only, no rng, no I/O
from itertools import product

GRID_N = 20
K = 3

N_BASE = 40
N_POIS = 40
SPATIAL_EXPR_VAR = 0.2
GAUSS_LENGTHSCALE = 2.0

N_REP = 5000
TAU = 0.5


XI_INT_SCALE = 1.5 

SCENARIOS = [
    ("nonspatial", "strict_null", 0),
    ("nonspatial", "shared_null", 1),
    ("nonspatial", "single_interaction", 2),
    ("nonspatial", "two_interaction", 3),
    ("spatial", "strict_null", 4),
    ("spatial", "shared_null", 5),
    ("spatial", "single_interaction", 6),
    ("spatial", "two_interaction", 7),
]

ACTIVE_INTERACTIONS = {
    "strict_null": set(),
    "shared_null": set(),

    # (t, k) zero-indexed
    "single_interaction": {(0, 0)},
    "two_interaction": {(0, 0), (1, 1)},
}
SEED = 67
S = GRID_N * GRID_N 
DF_PER_AXIS = 2
ORTHONORMALIZE_BASIS = True




SPATIAL_EXPR_VAR_VALUES = [0.2, 0.5, 0.8]
TAU_VALUES = [0.1, 0.5, 1.0]
GAUSS_LENGTHSCALE_VALUES = [1.0, 3.0, 5.0]
XI_INT_SCALE_VALUES = [0.5, 1.0, 3.0]

CONFIGS = [
    {"config_id": i, "spatial_expr_var": e, "tau": t, "ell": ell, "xi_scale": x}
    for i, (e, t, ell, x) in enumerate(
        product(SPATIAL_EXPR_VAR_VALUES, TAU_VALUES, GAUSS_LENGTHSCALE_VALUES, XI_INT_SCALE_VALUES)
    )
]