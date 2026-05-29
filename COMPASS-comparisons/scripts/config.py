# config.py  -- constants only, no rng, no I/O
GRID_N = 20
K = 3
N_BASE = 40
N_POIS = 40
N_REP = 8
N_REP_SENS = 3
SCENARIOS = ["null", "single_interaction", "two_interaction"]
R_VALS = [2, 4, 8]
SEED = 67

S = GRID_N * GRID_N  # derived — computed, not declared as a knob
