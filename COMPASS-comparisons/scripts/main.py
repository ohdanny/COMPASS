import sys
import time
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import logging


from compass.inference import compass_fit_gene
from compass.splines import build_bspline_basis, build_basis_to_rank

from scripts.vizio import (
    plot_eta_decomposition,
    save_csv,
    plot_cell_type,
    plot_scenarios_grid,
    plot_null_qq,
    plot_power_bar
    )
from scripts.config import (
    N_BASE,
    N_POIS,
    N_REP,
    N_REP_SENS,
    SCENARIOS,
    R_VALS,
    GRID_N,
    K,
    S,
    SEED,
)
from scripts.sim import build_grid, build_W, build_truth, simulate_counts,compute_power,setup_pi_y_plots
# from scripts.checks import sanity_checks

OUT_DIR = (Path(__file__).parent / ".." / "results").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "run.log"



log = setup_loger("COMPASS Demo Simulation")
rng = np.random.default_rng(SEED)


def one_run(scenario, 
            X_default,
            B_default,
            W,
            coords,
            rep_id, 
            r_override=None, seed=0):
    t0 = time.time()

    rng = np.random.default_rng(seed)
    B = B_default if r_override is None else build_basis_to_rank(jnp.asarray(coords), r_override)
    r = B.shape[1]

    N = np.maximum(5,N_BASE + rng.poisson(N_POIS, size=S)).astype(np.int64)
    truth = build_truth(scenario=scenario, r=r, rep=rep_id, rng=rng,B=B, X_default=X_default)
    Y,Pi = simulate_counts(truth, X_default, B, N, rng)
    
    # log.info(f"  Simulated data in {t_sim:.2f} sec | N mean={N.mean():.1f} (min={N.min()}, max={N.max()})")    
    
    res = compass_fit_gene(jnp.asarray(Y), jnp.asarray(N), jnp.asarray(W),
                           jnp.asarray(B), run_full=True, n_starts=3, verbose=False)
    t_fit = time.time() - t0


    # log.info(f"  Fitted model in {t_fit:.2f} sec | conv_reduced={res.fit_reduced.success} conv_full={res.fit_full.success} ")

    assert res.fit_full and res.wald, "this is for my linter"

    Pi = np.asarray(res.fit_full.Pi)
    wald_Ws = [float(w.W) for w in res.wald.results]
    wald_ps = [float(w.pval) for w in res.wald.results]

    return dict(scenario=scenario, rep_id=rep_id, r=r, time_sec=t_fit,
                conv_reduced=int(bool(res.fit_reduced.success)),
                conv_full=int(bool(res.fit_full.success)),
                pi_min=float(Pi.min()), pi_max=float(Pi.max()),
                Q_omni=float(res.score_test.Q), df_omni=int(res.score_test.df),
                p_omni=float(res.score_test.pval),
                W_t1=wald_Ws[0], p_t1=wald_ps[0],
                W_t2=wald_Ws[1], p_t2=wald_ps[1])


if __name__ == "__main__":

    coords = build_grid(GRID_N)
    W = build_W(coords, rng)
    T = W.shape[1]

    log.info("Plotting cell_type_composition.png")

    plot_cell_type(
        values=W,
        grid_n=GRID_N,
        rows=1,
        cols=T,
        out=OUT_DIR / "cell_type_composition.png",
        log=log )


    B_default = build_bspline_basis(coords, df_per_axis=2, orthonormalize=True)
    r_default = B_default.shape[1]
    X_default = W[:, :-1]

    log.info(f"S={S} T={T} K={K} r={r_default} | W means {np.round(W.mean(0),3)}")    
    log.info("Plotting scenarios_pi_true.png and scenarios_Y_true.png")

    combined_pi, pi_titles, combined_props, prop_titles, eta_by_scenario = setup_pi_y_plots(X_default, B_default, r_default, rng)
    
    # Pass it straight into your existing plotting function
    plot_scenarios_grid(
        values=combined_pi,
        grid_n=GRID_N,
        ax_titles=pi_titles,
        value_label="$\pi_{sgk}$", # type: ignore
        main="True within-gene Isoform Proportions $\pi_{sgk}$ (Rows = scenario/gene, Cols = isoforms)", # type: ignore
        out=OUT_DIR / "scenarios_pi_true.png",
        log=log
    )

    plot_scenarios_grid(
        values=combined_props,
        grid_n=GRID_N,
        ax_titles=prop_titles,
        value_label="$Y_{sgk} / N_{sg}$",
        main="Observed Isoform Proportions $Y_{sgk} / N_{sg}$. (Rows = scenario/gene, Cols = isoforms)", # type: ignore
        out=OUT_DIR / "scenarios_Y_true.png",
        log=log
    )


    for sc, comps in eta_by_scenario.items():
        plot_eta_decomposition(comps, GRID_N, sc, OUT_DIR / f"eta_decomposition_{sc}.png", log)
    

    main_rows = []
    log.info(f"--- Main runs (r={r_default}, n_rep per scenario={N_REP}) ---")

    for i, sc in enumerate(SCENARIOS, 1):
        for rep in range(1, N_REP + 1):
            r = one_run(
                scenario=sc,
                X_default=X_default,
                B_default=B_default,
                W=W,
                coords=coords,
                rep_id=rep,
                r_override=None,
                seed=10000 + 10000*i+rep # 10 * i + rep,
            )
            log.info(
                f"[{sc:<22} rep={rep}] "
                f"Q={r['Q_omni']:>7.2f}  "
                f"df={r['df_omni']}  "
                f"p={r['p_omni']:>9.3g}  "
                f"W1={r['W_t1']:>6.2f}  "
                f"p1={r['p_t1']:>9.3g}  "
                f"W2={r['W_t2']:>6.2f}  "
                f"p2={r['p_t2']:>9.3g}  "
                f"conv={r['conv_full']}/{r['conv_reduced']}  "
                f"pi=[{r['pi_min']:g},{r['pi_max']:g}]  "
                f"time={r['time_sec']:.1f}s"
                        )
            main_rows.append(r)

    log.info(f"--- r-sensitivity sweep (r in {{{','.join(map(str, R_VALS))}}}, n_rep={N_REP_SENS}) ---")
    sens_rows = []
    for i, sc in enumerate(SCENARIOS, 1):
        for rv in R_VALS:
            for rep in range(1, N_REP_SENS + 1):
                r = one_run(
                    scenario=sc,
                    X_default=X_default,
                    B_default=B_default,
                    W=W,
                    coords=coords,
                    rep_id=rep,
                    r_override=rv,
                    seed=20000 + 100*rv + rep #10 * i + rep,
                )
                sens_rows.append(r)
                # Sensitivity runs format
                log.info(
                    f"[r-sens {sc:<19} r={rv} rep={rep}] "
                    f"Q={r['Q_omni']:>7.2f} "
                    f"p={r['p_omni']:>9.3g} "
                    f"conv={r['conv_full']}/{r['conv_reduced']}  "
                    f"time={r['time_sec']:.1f}s"
                )


    results = sanity_checks(main_rows, sens_rows, log)


    # ---- null QQ plot (omnibus) ----

    plot_null_qq(np.array([r["p_omni"] for r in main_rows if r["scenario"] == "null"]),OUT_DIR / "null_qq_plot.png",log,)
        # ---- cell-type-specific power bar (single/two interaction) ----
    power_rows = compute_power(main_rows)
    plot_power_bar(power_rows, OUT_DIR / "cell_type_specific_power.png", log)


    # save results
    save_csv(power_rows, OUT_DIR / "cell_type_specific_power.csv", log) 
    save_csv([{"check": c, "pass": p} for c, p in results], OUT_DIR / "sanity_checks.csv", log)
    save_csv(main_rows,  OUT_DIR / "summary.csv", log)
    save_csv(sens_rows,  OUT_DIR / "r_sensitivity.csv", log)