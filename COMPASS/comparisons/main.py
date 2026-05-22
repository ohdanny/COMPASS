from pathlib import Path
import jax.numpy as jnp

import numpy as np
from compass import build_bspline_basis
from compass import setup_logger


from config import (SCENARIOS,GRID_N, K, S, N_REP,N_POIS, SEED,DF_PER_AXIS, ORTHONORMALIZE_BASIS)
from sim import build_grid, build_W,simulate_counts,build_regions,load_true_params

from concurrent.futures import ProcessPoolExecutor, as_completed
from methods import CompassMethod, SPVCMethod, SplisosmMethod

# from runone import compass_rep,spvc_rep
METHODS = {"compass": CompassMethod(), "spvc": SPVCMethod(), "splisosm": SplisosmMethod(),}
OUT_DIR = (Path(__file__).parent / "." / "res").resolve()
LOG_DIR = (Path(__file__).parent / "." / "logs").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def logname(method, expr, iso):
    return f"{expr}_{iso}_{method}.log"

def outname(expr, iso):
    outdir = OUT_DIR / expr / iso
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def make_seed(scenario_id,rep_id):
    return abs(hash((scenario_id,rep_id,SEED))) % (2 ** 32)

log = setup_logger("main", LOG_DIR / "main.log")
(LOG_DIR / "main.log").write_text("")


def run_method_scenario(method_name, scenario, Ys, Ns, W, B, coords, log, **kwargs):
    method = METHODS[method_name]
    method.init_worker(log, scenario) 


    for rep_id, (Y,N) in enumerate(zip(Ys, Ns), start=1):
        
        if method_name == "compass":
            method.run_rep(rep_id, Y=jnp.asarray(Y), N=jnp.asarray(N), W=W, B=B, **kwargs)
        elif method_name == "spvc":
            method.run_rep(rep_id, Y=Y, X=W[:,:-1], coords=coords, **kwargs)
        elif method_name == "splisosm":
            method.run_rep(rep_id, Y=Y, X=W[:, :-1], coords=coords, **kwargs)
        else:
            raise ValueError(f"Unknown method {method_name}")



if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    coords = build_grid(GRID_N)
    W = build_W(coords, rng)
    R = build_regions(coords, n_regions=3)
    B = build_bspline_basis(
        jnp.asarray(coords),
        df_per_axis=DF_PER_AXIS,
        orthonormalize=ORTHONORMALIZE_BASIS,
    )
    T = W.shape[1]
    X = W[:, :-1]

    true_params = load_true_params(T=T, r=B.shape[1], K=K)

    log.info(f"S={S}, T={T}, K={K}, r={B.shape[1]}")

    methods_map = dict()
    for expr, iso,scenario_id in SCENARIOS:
        scenario_out = outname(expr, iso)
        compass_log = setup_logger("compass", LOG_DIR / logname("compass", expr, iso))
        spvc_log = setup_logger("spvc", LOG_DIR / logname("spvc", expr, iso))
        splisosm_log = setup_logger("splisosm", LOG_DIR / logname("splisosm", expr, iso))
        compass_out = scenario_out / "compass_results.pkl"
        spvc_out = scenario_out / "spvc_results.rds"
        splisosm_out = scenario_out / "splisosm_results.pkl"

        Ys, Pis, Ns = [],[],[]

        for rep in range(1,N_REP+1):
            seed = make_seed(scenario_id,rep)
            rng = np.random.default_rng(seed)
            Y, Pi, N = simulate_counts(
                a_g=np.log(N_POIS),
                R=R,
                X=X,
                B=B,
                expr_setting=expr,
                iso_setting=iso,
                rep_id=rep,
                rng=rng,
                params=true_params
            )
            Ys.append(Y);  Pis.append(Pi); Ns.append(N) #type: ignore  # noqa: E702

        run_compass_full = "null" not in iso
        run_method_scenario("compass", (scenario_id, expr, iso), Ys, Ns, W, B, coords, compass_log, run_full=run_compass_full)
        run_method_scenario("spvc", (scenario_id, expr, iso), Ys, Ns, W, B, coords, spvc_log,n_cores=16)
        run_method_scenario("splisosm", (scenario_id, expr, iso), Ys, Ns, W, B, coords, splisosm_log)

        METHODS["compass"].save_scenario(compass_out)
        METHODS["spvc"].save_scenario(spvc_out)
        METHODS["splisosm"].save_scenario(splisosm_out)
        
        
