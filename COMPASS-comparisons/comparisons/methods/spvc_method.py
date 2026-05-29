import time
import traceback
from .base import Method
import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri  # , pandas2ri
from rpy2.robjects.packages import importr
import numpy as np
from pathlib import Path

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from .utils import r_to_python

jsonpickle_numpy.register_handlers()
np_cv = default_converter + numpy2ri.converter



class SPVCMethod(Method):
    name = "spvc"

    def init_worker(self, log, scenario,out_path,clear_existing: bool=False,raise_on_err: bool=True) -> None:
        self.expr,self.iso,self.scenario_id = scenario
        
        self.log = log
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.raise_on_err = raise_on_err

        if clear_existing and self.out_path.exists():
            self.out_path.unlink()
            if self._index_path().exists():
                self._index_path().unlink()
                
        self.done_reps = self._load_done_reps()

        # gets rid of R msgs in logs
        ro.r("sink('/dev/null')")

        # idek if i need these
        importr("spVC")
        importr("BPST")
        importr("Triangulation")
        build_triangulation()

    def run_rep(self,rep_id,**args):

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
            res = run_spvc_per_gene(**args)
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


        with np_cv.context():
                res_py = r_to_python(res)
        res_blob = jsonpickle.encode(res_py, unpicklable=True)

        record = {
            **base_record,
            "ok": True,
            "result": res_blob,
            "time_sec": t_fit,
        }  
        self._append(record, rep_id, ok=True)


        self.log.info(
            f"[{self.scenario_id} {self.expr:<8} {self.iso:<7} rep={rep_id:>2}] "
            f"│ t={t_fit:.1f}s"
        )
        return res
    
    
def build_triangulation():
    ro.r("""
    boundary <- matrix(c(0,0, 1,0, 1,1, 0,1), ncol=2, byrow=TRUE)
    Tr.cell <- Triangulation::TriMesh(boundary, n=2)
    V <<- as.matrix(Tr.cell$V)
    Tr <<- as.matrix(Tr.cell$Tr)
    """)


def run_spvc_per_gene(Y, X, coords, n_cores=2, min_nonzero=5, min_spot_counts=0):
    with np_cv.context():
        ro.globalenv["Y_r"] = np.asarray(Y).T
        ro.globalenv["X_r"] = np.asarray(X)
        ro.globalenv["S_r"] = np.asarray(coords)

        ro.r("""
        print(dim(Y_r))
        print(dim(X_r))
        print(dim(S_r))
        """)

        ro.r(f"""
        rownames(Y_r) <- paste0("iso", 1:nrow(Y_r))
        res <- spVC::test.spVC(Y = Y_r, X = X_r, S = S_r, V = V, Tr = Tr,
                               para.cores = {n_cores},
                               filter.min.nonzero = {min_nonzero},
                               filter.spot.counts = {min_spot_counts})
        """)
        # ro.r("print(str(res, max.level=3))")
        res = ro.globalenv["res"]

    return res
