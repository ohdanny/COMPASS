import pickle
from pathlib import Path
import csv

HERE     = Path(__file__).parent
RES_DIR  = (HERE.parent / "res").resolve()
OUT_PATH = HERE / "compass_pvals.csv"

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
        w.writerow(["scenario_id", "expr", "iso", "rep", "level", "t", "k", "pval", "pval_holm"])
        for sid, expr, iso in SCENARIOS:
            path = RES_DIR / expr / iso / "compass_results.pkl"
            if not path.exists():
                print(f"missing: {path}"); continue
            with path.open("rb") as fh:
                results = pickle.load(fh)
            for rep_id, r in enumerate(results, start=1):
                w.writerow([sid, expr, iso, rep_id, "omnibus", "", "",
                            float(r.score_test.pval), ""])
                if r.wald is not None:
                    for one in r.wald.results:
                        w.writerow([sid, expr, iso, rep_id, "cell",
                                    int(one.t), "", float(one.pval),
                                    "" if one.pval_holm is None else float(one.pval_holm)])
                if r.wald_by_cat is not None:
                    for one in r.wald_by_cat.per_pair:
                        if not one.eligible:
                            continue
                        w.writerow([sid, expr, iso, rep_id, "pair",
                                    int(one.t), int(one.k), float(one.pval),
                                    "" if one.p_holm is None else float(one.p_holm)])
            print(f"read scenario {sid} {expr}/{iso} ({len(results)} reps)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
