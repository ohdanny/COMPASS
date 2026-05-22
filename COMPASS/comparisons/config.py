# config.py  -- constants only, no rng, no I/O

GRID_N = 20
K = 3

N_BASE = 40
N_POIS = 40
N_REP = 100

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

