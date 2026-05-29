import pickle
from pathlib import Path
import csv

HERE     = Path(__file__).parent
RES_DIR  = (HERE.parent / "res").resolve()
OUT_PATH = HERE / "splisosm_pvals.csv"

SCENARIOS = [
    (0, "nonspatial", "strict_null"),
    (1, "nonspatial", "shared_null"),
    (2, "nonspatial", "single_interaction"),
    (3, "nonspatial", "two_interaction"),
    (4, "spatial",    "strict_null"),
    (5, "spatial",    "shared_null"),
    (6, "spatial",    "single_interaction"),
    (7, "spatial",    "two_interaction"),
]


def main():
    with OUT_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "expr", "iso", "rep", "test", "stat", "pval"])
        for sid, expr, iso in SCENARIOS:
            path = RES_DIR / expr / iso / "splisosm_results.pkl"
            if not path.exists():
                print(f"missing: {path}"); continue
            with path.open("rb") as fh:
                results = pickle.load(fh)
            for r in results:
                w.writerow([sid, expr, iso, r["rep_id"], "hsic_ir",
                            float(r["ir_stat"]), float(r["ir_pvalue"])])
                w.writerow([sid, expr, iso, r["rep_id"], "hsic_ic",
                            float(r["ic_stat"]), float(r["ic_pvalue"])])
            print(f"read scenario {sid} {expr}/{iso} ({len(results)} reps)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
