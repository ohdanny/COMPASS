import time
from .base import Method
from compass import compass_fit_gene
import numpy as np
import pickle

class CompassMethod(Method):

    name = "compass"

    def init_worker(self, log, scenario) -> None:
        self.log = log
        self.scenario_id,self.expr,self.iso = scenario
        self.results = []

    def run_rep(self, rep_id, **args):
        t0 = time.time()
        res = compass_fit_gene(**args)
        t_fit = time.time() - t0

        omnitbus = res.score_test
        Pi = np.asarray(res.fit_full.Pi) if res.fit_full else None
        pi_min = float(Pi.min()) if Pi is not None else 67
        pi_max = float(Pi.max()) if Pi is not None else 67
        conv_red = int(bool(res.fit_reduced.success))
        conv_full = int(bool(res.fit_full.success)) if res.fit_full else 67

        wald_ps = [float(w.pval) for w in res.wald.results] if res.wald else []
        wald_ws = [float(w.W) for w in res.wald.results] if res.wald else []
        wald_p_str = (
            " ".join(f"p_t{i+1}={p:.3g}" for i, p in enumerate(wald_ps))
            if wald_ps
            else "n/a"
        )
        wald_W_str = (
            " ".join(f"W_t{i+1}={W:.2f}" for i, W in enumerate(wald_ws))
            if wald_ws
            else "n/a"
        )

        self.log.info(
            f"[{self.scenario_id} {self.expr:<8} {self.iso:<7} rep={rep_id:>2}] "
            f"OMNI: Q={omnitbus.Q:>6.2f} df={omnitbus.df:>2} p={omnitbus.pval:>9.3g} │ "
            f"WALD: {wald_p_str} {wald_W_str} │ "
            f"FIT: conv={conv_full}/{conv_red} pi=[{pi_min:.2f},{pi_max:.2f}] t={t_fit:.1f}s"
        )

        self.results.append(res)
        return res
    
    def save_scenario(self, out_path) -> None:
        with open(out_path, "wb") as f:
            pickle.dump(self.results, f)
        self.log.info(f"Saved COMPASS results for scenario {self.scenario_id} to {out_path}")