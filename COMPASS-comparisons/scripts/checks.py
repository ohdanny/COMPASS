import numpy as np
import logging

from config import R_VALS

def sanity_checks(main_rows, sens_rows,log:logging.Logger):
    def col(rows, n, sc=None):
        return np.array(
            [float(r[n]) for r in rows if sc is None or r["scenario"] == sc]
        )

    # check optimizer converges
    converge_check = bool(
        np.all(col(main_rows, "conv_reduced") == 1)
        and np.all(col(main_rows, "conv_full") == 1)
    )

    # check probabilities in (0,1)
    pi_val_check = bool(
        col(main_rows, "pi_min").min() > 1e-6
        and col(main_rows, "pi_max").max() < 1 - 1e-6
    )

    log.info(
        f"pi range: {col(main_rows, 'pi_min').min():.3g} - {col(main_rows, 'pi_max').max():.3g}"
    )

    # check omnibus test behaves as expected
    omnibus_check = bool(
        np.median(col(main_rows, "p_omni", "null")) > 0.05
        and np.median(col(main_rows, "p_omni", "single_interaction")) < 0.05
        and np.median(col(main_rows, "p_omni", "two_interaction")) < 0.05
    )

    def mW(sc, c):
        return float(np.mean(col(main_rows, c, sc)))

    # mW = lambda sc, c: float(np.mean(col(main_rows, c, sc)))
    # checks that the active isoform in single_interaction has the largest W, and that both interaction terms in two_interaction are larger than null
    wald_active_dominant = mW("single_interaction", "W_t1") > mW(
        "single_interaction", "W_t2"
    )
    # cehcks that in the two_interaction scenario, both W_t1 and W_t2 are larger than either null scenario W
    wald_two_interaction = mW("two_interaction", "W_t1") > max(
        mW("null", "W_t1"), mW("null", "W_t2")
    ) and mW("two_interaction", "W_t2") > max(mW("null", "W_t1"), mW("null", "W_t2"))

    def ok_r(rv):
        s = [r for r in sens_rows if r["r"] == rv]

        # md = lambda sc: np.median([r["p_omni"] for r in s if r["scenario"] == sc])
        def md(sc):
            return np.median(col(s, "p_omni", sc))

        return (
            all(r["conv_full"] == 1 for r in s)
            and md("null") > 0.05
            and md("single_interaction") < 0.10
            and md("two_interaction") < 0.10
        )

    r_sense_check = all(ok_r(rv) for rv in R_VALS)

    final = [
        ("optimizer_converges", converge_check),
        ("prob_in_0_1", pi_val_check),
        ("omnibus_null_vs_alt", omnibus_check),
        ("wald_active_dominant", wald_active_dominant and wald_two_interaction),
        ("r_sensitivity_stable", r_sense_check),
    ]

    for c, p in final:
        log.info(f"  {c}: {p}")
    log.info("Done.")

    return final
