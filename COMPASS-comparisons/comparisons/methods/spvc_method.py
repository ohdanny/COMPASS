import time
from .base import Method
import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri  # , pandas2ri
from rpy2.robjects.packages import importr
import numpy as np
np_cv = default_converter + numpy2ri.converter 
ro.r('sink("/dev/null")')



def build_triangulation():
    ro.r("""
    boundary <- matrix(c(0,0, 1,0, 1,1, 0,1), ncol=2, byrow=TRUE)
    Tr.cell <- Triangulation::TriMesh(boundary, n=2)
    V <<- as.matrix(Tr.cell$V)
    Tr <<- as.matrix(Tr.cell$Tr)
    """)


def run_spvc_per_gene(
    Y, X, coords, n_cores=2, min_nonzero=5, min_spot_counts=0
):
    with np_cv.context():
        ro.globalenv["Y_r"] = np.asarray(Y).T
        ro.globalenv["X_r"] = np.asarray(X)
        ro.globalenv["S_r"] = np.asarray(coords)
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

class SPVCMethod(Method):
    name = "spvc"

    def init_worker(self, log, scenario) -> None:
        self.scenario_id,self.expr,self.iso = scenario
        
        self.log = log

        # gets rid of R msgs in logs
        ro.r(f"scenario_{self.scenario_id} <- list()")  # init empty list in R
        # idek if i need these
        importr("spVC")
        importr("BPST")
        importr("Triangulation")
        build_triangulation()

    def run_rep(self,rep_id,**args):
        t0 = time.time()
        try:
            res = run_spvc_per_gene(**args)
        except Exception as e:
            self.log.error(f"Error occurred while running SPVC replica {rep_id}: {e}")
            raise
        t_fit = time.time() - t0

        ro.r(f'scenario_{self.scenario_id}[[length(scenario_{self.scenario_id}) + 1]] <- res')  # append res to the R list called scenario_{scenario_id}

        list_len = int(ro.r(f"length(scenario_{self.scenario_id})")[0])  # type: ignore

        self.log.info(
            f"[{self.scenario_id} {self.expr:<8} {self.iso:<7} rep={rep_id:>2}] "
            f"│ list_len={list_len} "
            f"│ t={t_fit:.1f}s"
        )
        return res

    def save_scenario(self,out_path) -> None:
        ro.r(f'saveRDS(scenario_{self.scenario_id}, file="{out_path}")')
        ro.r(f'rm(scenario_{self.scenario_id}); gc()')
        self.log.info(f"Saved SPVC results for scenario {self.scenario_id} to {out_path}")
