from pathlib import Path
import numpy as np
import pandas as pd

HERE  = Path(__file__).parent
ALPHA = 0.05

ACTIVE = {
    "strict_null":         set(),
    "shared_null":         set(),
    "single_interaction":  {(0, 0)},
    "two_interaction":     {(0, 0), (1, 1)},
}
TESTED_T = (0, 1)  # ref_ct = 2; cell types tested by both methods


def simes(pvals):
    p = np.asarray(pvals, dtype=float)
    p = p[~np.isnan(p)]
    if p.size == 0:
        return 1.0
    p.sort()
    n = p.size
    return float((p * n / np.arange(1, n + 1)).min())


def level1(compass, spvc, splisosm, n_rep):
    rows = []
    omni = compass[compass.level == "omnibus"]
    g = omni.groupby(["scenario_id", "expr", "iso"])["pval"].apply(
        lambda s: (s < ALPHA).mean()
    ).reset_index(name="reject_rate")
    g["method"] = "compass"
    rows.append(g)

    g0 = spvc[(spvc.model == "constant") & (spvc.term == "gamma_0")]
    per_rep = g0.groupby(["scenario_id", "expr", "iso", "rep"])["pval"].apply(simes).reset_index(name="pval")
    g = per_rep.groupby(["scenario_id", "expr", "iso"])["pval"].apply(
        lambda s: (s < ALPHA).mean()
    ).reset_index(name="reject_rate")
    g["method"] = "spvc"
    rows.append(g)

    if splisosm is not None:
        for test_name, label in [("hsic_ir", "splisosm_ir"), ("hsic_ic", "splisosm_ic")]:
            df = splisosm[splisosm.test == test_name]
            if df.empty:
                continue
            g = df.groupby(["scenario_id", "expr", "iso"])["pval"].apply(
                lambda s: (s < ALPHA).mean()
            ).reset_index(name="reject_rate")
            g["method"] = label
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def level2(compass, spvc, n_rep):
    rows = []
    cell = compass[compass.level == "cell"].copy()
    cell["reject"] = cell.pval < ALPHA
    g = cell.groupby(["scenario_id", "expr", "iso", "t"])["reject"].mean().reset_index(name="reject_rate")
    g["method"] = "compass"
    rows.append(g)

    for t in TESTED_T:
        term = f"gamma_X{t+1}"
        df = spvc[(spvc.model == "varying") & (spvc.term == term)]
        if df.empty:
            continue
        per_rep = df.groupby(["scenario_id", "expr", "iso", "rep"])["pval"].apply(simes).reset_index(name="pval")
        per_rep["t"] = t
        per_rep["reject"] = per_rep.pval < ALPHA
        rej = per_rep.groupby(["scenario_id", "expr", "iso", "t"])["reject"].sum().reset_index(name="n_reject")
        m = rej.merge(n_rep, on=["scenario_id", "expr", "iso"])
        m["reject_rate"] = m.n_reject / m.n_rep
        m["method"] = "spvc"
        rows.append(m[["scenario_id", "expr", "iso", "t", "reject_rate", "method"]])
    return pd.concat(rows, ignore_index=True)


def level3(compass, spvc, n_rep):
    rows = []
    pair = compass[compass.level == "pair"].copy()
    pair["reject"] = pair.pval < ALPHA
    g = pair.groupby(["scenario_id", "expr", "iso", "t", "k"])["reject"].mean().reset_index(name="reject_rate")
    g["method"] = "compass"
    rows.append(g)

    sp_pairs = []
    for t in TESTED_T:
        term = f"gamma_X{t+1}"
        df = spvc[(spvc.model == "varying") & (spvc.term == term)].copy()
        if df.empty:
            continue
        df["t"] = t
        sp_pairs.append(df[["scenario_id", "expr", "iso", "rep", "k", "t", "pval"]])
    sp = pd.concat(sp_pairs, ignore_index=True) if sp_pairs else None
    if sp is not None:
        sp["reject"] = sp.pval < ALPHA
        rej = sp.groupby(["scenario_id", "expr", "iso", "t", "k"])["reject"].sum().reset_index(name="n_reject")
        m = rej.merge(n_rep, on=["scenario_id", "expr", "iso"])
        m["reject_rate"] = m.n_reject / m.n_rep
        m["method"] = "spvc"
        rows.append(m[["scenario_id", "expr", "iso", "t", "k", "reject_rate", "method"]])

    return pd.concat(rows, ignore_index=True), pair, sp


def level3_quality(pair_compass, sp_spvc, n_rep):
    rows = []
    for (sid, expr, iso, rep), sub in pair_compass.groupby(["scenario_id", "expr", "iso", "rep"]):
        rej = {(int(r.t), int(r.k)) for r in sub.itertuples() if r.reject}
        active = ACTIVE[iso]
        union = rej | active
        jacc = len(rej & active) / len(union) if union else 1.0
        rows.append((sid, expr, iso, rep, "compass", jacc, rej == active))

    if sp_spvc is not None:
        # Reps that appear in spVC varying at all
        seen = set()
        for (sid, expr, iso, rep), sub in sp_spvc.groupby(["scenario_id", "expr", "iso", "rep"]):
            rej = {(int(r.t), int(r.k)) for r in sub.itertuples() if r.reject}
            active = ACTIVE[iso]
            union = rej | active
            jacc = len(rej & active) / len(union) if union else 1.0
            rows.append((sid, expr, iso, rep, "spvc", jacc, rej == active))
            seen.add((sid, expr, iso, rep))
        # Reps where spVC fit no varying isoforms → rejected set is empty
        for _, row in n_rep.iterrows():
            sid, expr, iso, n = row.scenario_id, row.expr, row.iso, row.n_rep
            active = ACTIVE[iso]
            jacc_empty = 0.0 if active else 1.0
            exact_empty = (set() == active)
            for rep in range(1, int(n) + 1):
                if (sid, expr, iso, rep) in seen:
                    continue
                rows.append((sid, expr, iso, rep, "spvc", jacc_empty, exact_empty))

    return pd.DataFrame(rows, columns=["scenario_id", "expr", "iso", "rep", "method", "jaccard", "exact"])


def main():
    compass = pd.read_csv(HERE / "compass_pvals.csv")
    spvc    = pd.read_csv(HERE / "spvc_pvals.csv")
    splisosm_path = HERE / "splisosm_pvals.csv"
    splisosm = pd.read_csv(splisosm_path) if splisosm_path.exists() else None
    n_rep   = compass[compass.level == "omnibus"].groupby(["scenario_id", "expr", "iso"])["rep"].nunique().reset_index(name="n_rep")

    pd.set_option("display.float_format", "{:.3f}".format)

    l1 = level1(compass, spvc, splisosm, n_rep)
    print(f"\n=== Level 1: global rejection rate (alpha={ALPHA}) ===")
    print(l1.pivot_table(index=["expr", "iso"], columns="method", values="reject_rate"))

    l2 = level2(compass, spvc, n_rep)
    print(f"\n=== Level 2: cell-type rejection rate (alpha={ALPHA}) ===")
    print(l2.pivot_table(index=["expr", "iso", "t"], columns="method", values="reject_rate"))

    l3, pair, sp = level3(compass, spvc, n_rep)
    print(f"\n=== Level 3: (t,k) pair rejection rate (alpha={ALPHA}) ===")
    print(l3.pivot_table(index=["expr", "iso", "t", "k"], columns="method", values="reject_rate"))

    q = level3_quality(pair, sp, n_rep)
    summary = q.groupby(["expr", "iso", "method"]).agg(
        mean_jaccard=("jaccard", "mean"),
        exact_recovery=("exact", "mean"),
    ).reset_index()
    print("\n=== Level 3 quality: per-rep Jaccard & exact-set recovery vs ACTIVE ===")
    print(summary.pivot_table(index=["expr", "iso"], columns="method", values=["mean_jaccard", "exact_recovery"]))


if __name__ == "__main__":
    main()
