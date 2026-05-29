from pathlib import Path
import time
from compass import compass_fit_gene
import numpy as np
import traceback

from .base import Method
from .utils import jax_to_numpy
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()

class CompassMethod(Method):

    name = "compass"

    def init_worker(self, log, scenario, out_path, clear_existing: bool=False, raise_on_err: bool=False) -> None:
        self.log = log
        self.expr,self.iso,self.scenario_id = scenario
        self.results = []
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if clear_existing and self.out_path.exists():
            self.out_path.unlink()
            if self._index_path().exists():
                self._index_path().unlink()
        

        self.done_reps = self._load_done_reps()
        self.raise_on_err = raise_on_err


    def run_rep(self, rep_id, **args):
        if rep_id in self.done_reps:
            self.log.info(
                f"[{self.scenario_id} {self.expr:<8} {self.iso:<7} rep={rep_id:>2}] "
                f"SKIP (already done)"
            )
            return None
        
        base_record = {
            "scenario_id": self.scenario_id,
            "expr": self.expr,
            "iso": self.iso,
            "rep_id": rep_id,
        }

        t0 = time.time()
        try:
            res = compass_fit_gene(**args)
        except Exception as e:
            t_err = time.time() - t0
            record = {
                **base_record,
                "ok": False,
                "error": str(e),
                "time_sec": t_err,
                "traceback": traceback.format_exc(),
            }
            self._append(record, rep_id, ok=False)
            self.log.error(
                f"[{self.scenario_id} {self.expr:<8} {self.iso:<7} rep={rep_id:>2}] "
                f"FAILED: {e!s} (t={t_err:.1f}s)"
            )
            if self.raise_on_err:
                raise
            return None

        t_fit = time.time() - t0

        omnitbus = res.score_test
        Pi = np.asarray(res.fit_full.Pi) if res.fit_full else None
        pi_min = float(Pi.min()) if Pi is not None else None
        pi_max = float(Pi.max()) if Pi is not None else None
        conv_red = int(bool(res.fit_reduced.success))
        conv_full = int(bool(res.fit_full.success)) if res.fit_full else None

        wald_ps = [float(w.pval) for w in res.wald.results] if res.wald else []
        wald_ws = [float(w.W) for w in res.wald.results] if res.wald else []


        res_np = jax_to_numpy(res)
        res_json = jsonpickle.encode(res_np,unpicklable=True)
        record = {
            **base_record,
            "ok": True,
            "result": res_json,
            "time_sec": t_fit
        }
        self._append(record, rep_id, ok=True)

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

        return res