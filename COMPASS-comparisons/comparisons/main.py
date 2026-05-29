from pathlib import Path
import numpy as np
from compass import build_bspline_basis
from compass import setup_logger
from argparse import ArgumentParser,Namespace
import os

from methods import CompassMethod, SPVCMethod  # , SplisosmMethod
from config import (SCENARIOS,GRID_N, K, S, N_REP,N_POIS, SEED,DF_PER_AXIS, ORTHONORMALIZE_BASIS,CONFIGS)
from sim import build_grid, build_W,simulate_counts,build_regions,load_true_params,guassian_kernel

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL) # turns off rlly annoying warning when using cpu
from jax import config   # noqa: E402
import jax.numpy as jnp  # noqa: E402


# we NEED 64 bit precision i had some weird numerical issues without it
# also it runs faster with 64 bit ???? idk why rn but ok
config.update("jax_enable_x64", True)

# from runone import compass_rep,spvc_rep
METHODS = {"compass": CompassMethod(), "spvc": SPVCMethod()} #, "splisosm": SplisosmMethod(),}
METHOD_KWARGS = {
    "compass": {"run_full": True},
    "spvc": {"n_cores": os.cpu_count()},
}

def logname(method, expr, iso):
    return f"{expr}_{iso}_{method}.log"

def outname(outdir,expr, iso):
    outdir = outdir / expr / iso
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def config_dirname(cfg):
    def fmt(x):
        return f"{x:g}".replace(".", "p")
    return f"TAU{fmt(cfg['tau'])}_ELL{fmt(cfg['ell'])}_EPS{fmt(cfg['spatial_expr_var'])}_XI{fmt(cfg['xi_scale'])}"

def make_seed(config_id,scenario_id,rep_id):
    return abs(hash((config_id, scenario_id, rep_id, SEED))) % (2 ** 32)


def run_method_scenario(method_name, scenario, out_path, Ys, Ns, W, B, coords, log, **kwargs):
    # init method with out_path so it can manage resuming and saving
    method = METHODS[method_name]
    method.init_worker(log, scenario, out_path) 


    for rep_id, (Y,N) in enumerate(zip(Ys, Ns), start=1):
        if method_name == "compass":
            method.run_rep(rep_id, Y=jnp.asarray(Y), N=jnp.asarray(N), W=W, B=B, **kwargs)
        elif method_name == "spvc":
            method.run_rep(rep_id, Y=Y, X=W[:,:-1], coords=coords, **kwargs)
        elif method_name == "splisosm":
            method.run_rep(rep_id, Y=Y, X=W[:, :-1], coords=coords, **kwargs)
        else:
            raise ValueError(f"Unknown method {method_name}")



def get_scenario(task_id):
    for scenario in SCENARIOS:
        if scenario[2] == task_id:
            return scenario
    raise ValueError(f"Invalid task_id {task_id}")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=[scenario[2] for scenario in SCENARIOS], required=True)
    parser.add_argument("--method", type=str, choices=METHODS.keys(), required=True)
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Whether to clear existing results for this scenario and method (if any) before running",
    )
    parser.add_argument(
        "--no-raise-on-error",
        action="store_true",
        help="Whether to NOT raise an error if existing results are found for this scenario and method during simulation. Useful if want to just run as much as possible even if some fail",
    )
    
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Directory to save logs"
    )
    parser.add_argument(
        "--config-id", type=int, default=0, help="ID of the configuration to run"
    )

    return parser.parse_args()
    
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    args = parse_args()
    scenario, method_name = get_scenario(args.task_id), args.method
    expr, iso, scenario_id = scenario

    if not 0 <= args.config_id < len(CONFIGS):
        raise ValueError(f"config-id {args.config_id} out of range [0, {len(CONFIGS)})")
    cfg = CONFIGS[args.config_id]
    cfg_tag = config_dirname(cfg)

    outdir = Path(args.output_dir) / cfg_tag
    logdir = Path(args.log_dir) / cfg_tag
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

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
    kernel_cached = guassian_kernel(coords, lengthscale=cfg['ell'])
    kernel_cached *= cfg['tau']
    true_params = load_true_params(T=T, r=B.shape[1], K=K, xi_int_scale=cfg['xi_scale'])

    log = setup_logger(method_name, logdir / logname(method_name, expr, iso))
    log.info(f"S={S}, T={T}, K={K}, r={B.shape[1]}")

    scenario_out = outname(outdir, expr, iso)
    out_path = scenario_out / f"{method_name}_results.jsonl"

    Ys, Pis, Ns = [],[],[]

    for rep in range(1,N_REP+1):
        seed = make_seed(args.config_id,scenario_id,rep)
        rng = np.random.default_rng(seed)
        Y, Pi, N = simulate_counts(
            a_g=np.log(N_POIS),
            R=R,
            X=X,
            B=B,
            K=kernel_cached,
            expr_setting=expr,
            iso_setting=iso,
            rep_id=rep,
            rng=rng,
            params=true_params,
            spatial_expr_var=cfg['spatial_expr_var'],
        )
        Ys.append(Y);  Pis.append(Pi); Ns.append(N) #type: ignore  # noqa: E702

    extra_kwargs = METHOD_KWARGS.get(method_name, {})
    run_method_scenario(method_name, scenario, out_path, Ys, Ns, W, B, coords, log, **extra_kwargs)