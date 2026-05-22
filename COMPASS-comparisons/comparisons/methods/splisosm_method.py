import time
import pickle

import numpy as np
import pandas as pd
import anndata as ad

from .base import Method
from splisosm import SplisosmNP


def run_splisosm_per_gene(Y, X, coords, min_counts=1, min_bin_pct=0.0):
    """
    Run SPLISOSM's spatial variability tests on a single simulated gene.

    Parameters
    ----------
    Y : array (S, K)
        Isoform counts per spot. S spots, K isoform categories.
    X : array (S, T-1)
        Cell-type composition (reference category dropped). SPLISOSM's
        unconditional SV tests do not use X, but it is accepted to match
        the calling convention of the other methods. Used only for the
        differential-usage test below.
    coords : array (S, 2)
        Spatial coordinates of each spot.

    Returns
    -------
    dict
        Test statistics and p-values for HSIC-IR (isoform usage ratio) and
        HSIC-IC (isoform expression).
    """
    Y = np.asarray(Y)
    coords = np.asarray(coords)
    S, K = Y.shape

    # Wrap counts as AnnData: a single gene with K isoforms.
    var_df = pd.DataFrame({
        "gene_symbol": ["gene1"] * K,
        "transcript_name": [f"iso{k + 1}" for k in range(K)],
    })
    obs_df = pd.DataFrame({
        "spot_id": [f"spot_{s}" for s in range(S)],
    })

    adata = ad.AnnData(X=Y.astype(np.float32), obs=obs_df, var=var_df)
    adata.var_names = var_df["transcript_name"].values
    adata.var_names_make_unique()
    adata.obs_names = obs_df["spot_id"].values
    adata.obsm["spatial"] = coords
    adata.layers["counts"] = adata.X.copy()

    model = SplisosmNP()
    model.setup_data(
        adata=adata,
        spatial_key="spatial",
        layer="counts",
        group_iso_by="gene_symbol",
        gene_names="gene_symbol",
        min_counts=min_counts,
        min_bin_pct=min_bin_pct,
        filter_single_iso_genes=False,
    )

    # HSIC-IR: spatially variable isoform usage ratio (the main SVP test)
    model.test_spatial_variability(
        method="hsic-ir",
        null_method="clt",
        ratio_transformation="none",
        nan_filling="mean",
    )
    res_ir = model.get_formatted_test_results(test_type="sv")

    # HSIC-IC: spatially variable isoform expression (multivariate counts)
    model.test_spatial_variability(method="hsic-ic", null_method="clt")
    res_ic = model.get_formatted_test_results(test_type="sv")

    return {
        "ir_stat": float(res_ir["statistic"].iloc[0]),
        "ir_pvalue": float(res_ir["pvalue"].iloc[0]),
        "ic_stat": float(res_ic["statistic"].iloc[0]),
        "ic_pvalue": float(res_ic["pvalue"].iloc[0]),
    }


class SplisosmMethod(Method):
    name = "splisosm"

    def init_worker(self, log, scenario) -> None:
        self.log = log
        self.scenario_id, self.expr, self.iso = scenario
        self.results = []

    def run_rep(self, rep_id, Y, X, coords, **kwargs):
        t0 = time.time()
        try:
            res = run_splisosm_per_gene(Y=Y, X=X, coords=coords)
        except Exception as e:
            self.log.error(
                f"Error running SPLISOSM replicate {rep_id}: {e}"
            )
            raise
        t_fit = time.time() - t0

        rec = {
            "rep_id": rep_id,
            "scenario_id": self.scenario_id,
            "expr": self.expr,
            "iso": self.iso,
            "ir_stat": res["ir_stat"],
            "ir_pvalue": res["ir_pvalue"],
            "ic_stat": res["ic_stat"],
            "ic_pvalue": res["ic_pvalue"],
            "time_sec": t_fit,
        }
        self.results.append(rec)

        self.log.info(
            f"[{self.scenario_id} {self.expr:<8} {self.iso:<7} rep={rep_id:>2}] "
            f"HSIC-IR: stat={res['ir_stat']:>6.3f} p={res['ir_pvalue']:>9.3g} │ "
            f"HSIC-IC: stat={res['ic_stat']:>6.3f} p={res['ic_pvalue']:>9.3g} │ "
            f"t={t_fit:.1f}s"
        )
        return rec

    def save_scenario(self, out_path) -> None:
        with open(out_path, "wb") as f:
            pickle.dump(self.results, f)
        self.log.info(
            f"Saved SPLISOSM results for scenario {self.scenario_id} "
            f"to {out_path}"
        )
