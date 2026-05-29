from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
HERE  = Path(__file__).parent
ALPHA = 0.05
BH_Q  = 0.05
ACTIVE = {
    "strict_null":         set(),
    "shared_null":         set(),
    "single_interaction":  {(0, 0)},
    "two_interaction":     {(0, 0), (1, 1)},
}
TESTED_T = (0, 1)               # ref_ct = 2; both methods test these
TESTED_K = (0, 1)               # k = 2 is reference category
TESTED_PAIRS = [(t, k) for t in TESTED_T for k in TESTED_K]
# ---------- helpers ----------
def simes(pvals):
    p = np.asarray(pvals, dtype=float)
    p = p[~np.isnan(p)]
    if p.size == 0:
        return 1.0
    p.sort()
    n = p.size
    return float((p * n / np.arange(1, n + 1)).min())

def regime_scenario(iso):
    return "null" if not ACTIVE[iso] else "alt"
def regime_cell(iso, t):
    return "power" if any(at == t for (at, _) in ACTIVE[iso]) else "type-I"
def regime_pair(iso, t, k):
    return "power" if (t, k) in ACTIVE[iso] else "type-I"
def fmt(x):
    return "  NaN" if pd.isna(x) else f"{x:5.3f}"
def section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
# ---------- L1: omnibus / global ----------
def build_l1(compass, spvc, n_rep):
    # compass: per-rep omnibus pval
    c = compass[compass.level == "omnibus"][["scenario_id","expr","iso","rep","pval"]].copy()
    c["method"] = "compass"
    # spvc: Simes of gamma_0 (constant) across isoforms per rep
    g0 = spvc[(spvc.model == "constant") & (spvc.term == "gamma_0")]
    s = g0.groupby(["scenario_id","expr","iso","rep"])["pval"].apply(simes).reset_index()
    s["method"] = "spvc"
    return pd.concat([c, s], ignore_index=True)
def report_l1(l1):
    section("LEVEL 1 — Global gene-level detection (alpha=%.2f, BH q=%.2f)" % (ALPHA, BH_Q))
    print("Methods: COMPASS omnibus  |  spVC (Simes across isoforms of gamma_0)")
    print("Metrics: empirical rejection rate, type-I error, power, FDR")
    l1 = l1.copy()
    l1["regime"] = l1.iso.map(regime_scenario)
    l1["reject"] = l1.pval < ALPHA
    rates = (l1.groupby(["regime","expr","iso","method"])["reject"].mean()
                .unstack("method").reset_index())
    print("\nEmpirical rejection rate per scenario:")
    print(rates.to_string(index=False, formatters={c: fmt for c in rates.columns if c not in ("regime","expr","iso")}))
    # type I error (null scenarios) and power (alt scenarios)
    summary = (l1.groupby(["regime","method"])["reject"].mean()
                  .unstack("method"))
    print("\nMean across regime:")
    if "null" in summary.index:
        print(f"  Type-I error (nulls)  : compass={fmt(summary.loc['null','compass'])}  spvc={fmt(summary.loc['null','spvc'])}")
    if "alt" in summary.index:
        print(f"  Power        (alts)   : compass={fmt(summary.loc['alt','compass'])}  spvc={fmt(summary.loc['alt','spvc'])}")
    # Realized FDR & power at BH q=0.05, pooled across all reps + all scenarios
    print(f"\nPooled BH at q={BH_Q} (each rep treated as one gene):")
    for m in ("compass", "spvc"):
        d = l1[l1.method == m].copy()
        p = d.pval.to_numpy()
        is_alt = d.iso.map(lambda i: bool(ACTIVE[i])).to_numpy()
        rej = multipletests(p, alpha=BH_Q, method="fdr_bh")[0]
        n_rej = int(rej.sum())
        n_false = int(((~is_alt) & rej).sum())
        n_true_pos = int(is_alt.sum())
        fdr  = n_false / n_rej if n_rej else 0.0
        pwr  = (is_alt & rej).sum() / n_true_pos if n_true_pos else float("nan")
        print(f"  {m:7s}: rejections={n_rej:4d}  FDR={fdr:.3f}  power={pwr:.3f}")
# ---------- L2: cell-type localization ----------
def build_l2(compass, spvc, n_rep):
    rows = []
    # compass: wald cell-type pval (None for null scenarios)
    c = compass[compass.level == "cell"][["scenario_id","expr","iso","rep","t","pval"]].copy()
    c["t"] = c["t"].astype(int); c["method"] = "compass"
    rows.append(c)
    # spvc: Simes(gamma_X{t+1}) across isoforms per rep, per cell type t
    for t in TESTED_T:
        df = spvc[(spvc.model == "varying") & (spvc.term == f"gamma_X{t+1}")]
        if df.empty:
            continue
        per_rep = df.groupby(["scenario_id","expr","iso","rep"])["pval"].apply(simes).reset_index()
        per_rep["t"] = t; per_rep["method"] = "spvc"
        rows.append(per_rep)
    out = pd.concat(rows, ignore_index=True)
    # for reps with no spvc varying entries for this t, treat pval=NaN → non-reject
    return out
def report_l2(l2, compass, n_rep):
    section("LEVEL 2 — Cell-type localization (alpha=%.2f)" % ALPHA)
    print("Methods: COMPASS wald (per cell type)  |  spVC Simes(gamma_X{t+1}) across isoforms")
    print("Metrics: rejection rate, TPR (active cells), FPR (inactive cells), active-cell-type recovery")
    l2 = l2.copy()
    l2["regime"] = l2.apply(lambda r: regime_cell(r.iso, int(r.t)), axis=1)
    l2["reject"] = l2.pval < ALPHA
    # need to normalize spvc rejections by N_REP (reps without varying = non-reject)
    rates_c = (l2[l2.method == "compass"]
               .groupby(["regime","expr","iso","t"])["reject"].mean().reset_index(name="compass"))
    spvc_rej = (l2[l2.method == "spvc"].groupby(["expr","iso","t"])["reject"].sum().reset_index(name="n_reject"))
    spvc_rej = spvc_rej.merge(n_rep, on=["expr","iso"])
    spvc_rej["spvc"] = spvc_rej.n_reject / spvc_rej.n_rep
    spvc_rej["regime"] = spvc_rej.apply(lambda r: regime_cell(r.iso, int(r.t)), axis=1)
    rates = rates_c.merge(spvc_rej[["regime","expr","iso","t","spvc"]],
                          on=["regime","expr","iso","t"], how="outer")
    rates = rates.sort_values(["regime","expr","iso","t"])
    print("\nRejection rate per (scenario, cell type):")
    print(rates.to_string(index=False, formatters={"compass": fmt, "spvc": fmt}))
    # TPR / FPR aggregated
    print("\nPooled cell-type metrics:")
    for m in ("compass", "spvc"):
        # use raw l2 rows (spvc misses-as-non-reject handled separately)
        if m == "compass":
            d = l2[l2.method == "compass"]
            if d.empty: print(f"  {m:7s}: (no data — null scenarios skipped wald)"); continue
            tpr = d[d.regime == "power"].reject.mean() if (d.regime == "power").any() else float("nan")
            fpr = d[d.regime == "type-I"].reject.mean() if (d.regime == "type-I").any() else float("nan")
        else:
            d = l2[l2.method == "spvc"]
            # treat missing-rep as non-reject
            ds = d.assign(reject=d.reject.fillna(False))
            grp_keys = ["expr","iso","t"]
            full = pd.MultiIndex.from_product(
                [n_rep[["expr","iso"]].drop_duplicates().itertuples(index=False), TESTED_T],
                names=["scen", "t"]
            )
            n_total = sum(int(r.n_rep) * len(TESTED_T) for _, r in n_rep.iterrows())
            n_rej = int(ds.reject.sum())
            # TPR / FPR via (scenario, t)-level regime
            ds = ds.assign(regime=ds.apply(lambda r: regime_cell(r.iso, int(r.t)), axis=1))
            # power rows = reps where regime=power; FPR rows = reps where regime=type-I
            # For full denominator, include all reps × all t in that regime
            denoms = {"power": 0, "type-I": 0}
            nums   = {"power": 0, "type-I": 0}
            for _, row in n_rep.iterrows():
                for t in TESTED_T:
                    reg = regime_cell(row.iso, t)
                    denoms[reg] += int(row.n_rep)
                    nums[reg]   += int(((ds.expr == row.expr) & (ds.iso == row.iso) & (ds.t == t) & ds.reject).sum())
            tpr = nums["power"] / denoms["power"] if denoms["power"] else float("nan")
            fpr = nums["type-I"] / denoms["type-I"] if denoms["type-I"] else float("nan")
        print(f"  {m:7s}: TPR={fmt(tpr)}  FPR={fmt(fpr)}")
    # active-cell-type set recovery (alt scenarios only, per-rep exact match of rejected cells vs active cells)
    print("\nActive cell-type recovery (alt scenarios, per-rep exact match):")
    alt_isos = [i for i in ACTIVE if ACTIVE[i]]
    for m in ("compass", "spvc"):
        d = l2[(l2.method == m) & (l2.iso.isin(alt_isos))]
        if d.empty:
            print(f"  {m:7s}: (no data)"); continue
        # need to also fill missing spvc reps (treated as not rejecting any t)
        per_rep = d.assign(reject=d.reject.fillna(False)).groupby(
            ["expr","iso","rep"]).apply(
            lambda g: set(int(r.t) for r in g.itertuples() if r.reject),
            include_groups=False
        )
        # active cells per iso
        match = per_rep.reset_index().apply(
            lambda r: r[0] == {at for (at, _) in ACTIVE[r.iso]}, axis=1
        )
        # account for reps where spvc had no entry at all (treated as empty rejected set)
        # those reps simply don't appear in per_rep; for spvc, fill them in:
        if m == "spvc":
            n_seen = per_rep.groupby(level=["expr","iso"]).size().reset_index(name="n_seen")
            extra_match = 0; extra_total = 0
            for _, row in n_rep.iterrows():
                if row.iso not in alt_isos: continue
                seen = n_seen[(n_seen.expr == row.expr) & (n_seen.iso == row.iso)]
                n_unseen = int(row.n_rep) - (int(seen.n_seen.iloc[0]) if not seen.empty else 0)
                if n_unseen <= 0: continue
                # unseen reps reject ∅; matches iff active cells = ∅; for alt iso it's never ∅
                extra_total += n_unseen
            rec = (match.sum()) / (len(match) + extra_total)
        else:
            rec = match.mean()
        print(f"  {m:7s}: {fmt(rec)}")
# ---------- L3: (t,k) pair ----------
def build_l3(compass, spvc, n_rep):
    rows = []
    c = compass[compass.level == "pair"][["scenario_id","expr","iso","rep","t","k","pval"]].copy()
    c["t"] = c["t"].astype(int); c["k"] = c["k"].astype(int); c["method"] = "compass"
    rows.append(c)
    for t in TESTED_T:
        df = spvc[(spvc.model == "varying") & (spvc.term == f"gamma_X{t+1}")].copy()
        if df.empty:
            continue
        df["t"] = t; df["k"] = df["k"].astype(int); df["method"] = "spvc"
        rows.append(df[["scenario_id","expr","iso","rep","t","k","pval","method"]])
    return pd.concat(rows, ignore_index=True)
def report_l3(l3, n_rep):
    section("LEVEL 3 — (t,k) pair localization (alpha=%.2f, BH q=%.2f)" % (ALPHA, BH_Q))
    print("Methods: COMPASS wald_by_cat  |  spVC gamma_X{t+1} from iso k's varying model")
    print("Metrics: rejection rate, power (active pairs), FDR, exact-set recovery, mean Jaccard")
    l3 = l3.copy()
    l3["regime"] = l3.apply(lambda r: regime_pair(r.iso, int(r.t), int(r.k)), axis=1)
    l3["reject"] = l3.pval < ALPHA
    # per-(scenario, t, k) rejection rate. spvc misses normalized to N_REP.
    rc = (l3[l3.method == "compass"]
          .groupby(["regime","expr","iso","t","k"])["reject"].mean().reset_index(name="compass"))
    rs = l3[l3.method == "spvc"].groupby(["expr","iso","t","k"])["reject"].sum().reset_index(name="n_reject")
    rs = rs.merge(n_rep, on=["expr","iso"])
    rs["spvc"] = rs.n_reject / rs.n_rep
    rs["regime"] = rs.apply(lambda r: regime_pair(r.iso, int(r.t), int(r.k)), axis=1)
    rates = rc.merge(rs[["regime","expr","iso","t","k","spvc"]],
                     on=["regime","expr","iso","t","k"], how="outer")
    rates = rates.sort_values(["regime","expr","iso","t","k"])
    print("\nRejection rate per (scenario, t, k):")
    print(rates.to_string(index=False, formatters={"compass": fmt, "spvc": fmt}))
    # Pair-level power and FPR (per-pair under nulls / inactive within alt)
    print("\nPooled pair-level metrics:")
    for m in ("compass", "spvc"):
        d = l3[l3.method == m]
        if d.empty: print(f"  {m:7s}: (no data)"); continue
        if m == "compass":
            tpr = d[d.regime == "power"].reject.mean() if (d.regime == "power").any() else float("nan")
            fpr = d[d.regime == "type-I"].reject.mean() if (d.regime == "type-I").any() else float("nan")
        else:
            # normalize by full denominator of (reps × tested pairs)
            denoms = {"power": 0, "type-I": 0}; nums = {"power": 0, "type-I": 0}
            for _, row in n_rep.iterrows():
                for (t, k) in TESTED_PAIRS:
                    reg = regime_pair(row.iso, t, k)
                    denoms[reg] += int(row.n_rep)
                    nums[reg]   += int(((d.expr == row.expr) & (d.iso == row.iso) &
                                        (d.t == t) & (d.k == k) & d.reject).sum())
            tpr = nums["power"] / denoms["power"] if denoms["power"] else float("nan")
            fpr = nums["type-I"] / denoms["type-I"] if denoms["type-I"] else float("nan")
        print(f"  {m:7s}: power={fmt(tpr)}  FPR(inactive pairs)={fmt(fpr)}")
    # Per-rep BH at q within rep across tested pairs → realized per-rep FDR; average over reps
    print(f"\nPer-rep BH at q={BH_Q} (across tested pairs in that rep), averaged across reps:")
    for m in ("compass", "spvc"):
        d = l3[l3.method == m]
        if d.empty: print(f"  {m:7s}: (no data)"); continue
        fdr_list, pwr_list = [], []
        for _, n in n_rep.iterrows():
            active = ACTIVE[n.iso]
            sub_scenario = d[(d.expr == n.expr) & (d.iso == n.iso)]
            for rep in range(1, int(n.n_rep) + 1):
                sub = sub_scenario[sub_scenario.rep == rep]
                if sub.empty:
                    # spvc rep with no varying pairs → no rejections
                    pwr_list.append(0.0 if active else float("nan"))
                    continue
                p = sub.pval.fillna(1.0).to_numpy()
                rej = multipletests(p, alpha=BH_Q, method="fdr_bh")[0]
                rej_pairs = {(int(sub.t.iloc[i]), int(sub.k.iloc[i])) for i in np.where(rej)[0]}
                tp = len(rej_pairs & active)
                fp = len(rej_pairs - active)
                fdr_list.append(fp / (tp + fp) if (tp + fp) else 0.0)
                if active:
                    pwr_list.append(tp / len(active))
        fdr = float(np.mean(fdr_list)) if fdr_list else float("nan")
        pwr = float(np.nanmean(pwr_list)) if pwr_list else float("nan")
        print(f"  {m:7s}: mean realized FDR={fmt(fdr)}  mean realized power={fmt(pwr)}")
    # Exact-set recovery & Jaccard (raw alpha rejection per rep)
    print("\nExact active-set recovery & Jaccard (alpha=%.2f raw, per rep):" % ALPHA)
    alt_isos = [i for i in ACTIVE if ACTIVE[i]]
    for m in ("compass", "spvc"):
        d = l3[l3.method == m]
        rows = []
        for _, n in n_rep.iterrows():
            active = ACTIVE[n.iso]
            sub_scenario = d[(d.expr == n.expr) & (d.iso == n.iso)]
            for rep in range(1, int(n.n_rep) + 1):
                sub = sub_scenario[sub_scenario.rep == rep]
                rej_pairs = {(int(r.t), int(r.k)) for r in sub.itertuples() if r.reject}
                union = rej_pairs | active
                jacc = len(rej_pairs & active) / len(union) if union else 1.0
                rows.append((n.expr, n.iso, jacc, rej_pairs == active))
        q = pd.DataFrame(rows, columns=["expr","iso","jacc","exact"])
        q["regime"] = q.iso.map(regime_scenario)
        # report split by null vs alt
        null_q = q[q.regime == "null"]
        alt_q  = q[q.regime == "alt"]
        print(f"  {m:7s}:")
        if not null_q.empty:
            print(f"    nulls : exact_recovery={fmt(null_q.exact.mean())}  jaccard={fmt(null_q.jacc.mean())}  (trivially 1.0 if no false rejections)")
        if not alt_q.empty:
            print(f"    alts  : exact_recovery={fmt(alt_q.exact.mean())}  jaccard={fmt(alt_q.jacc.mean())}")
# ---------- main ----------
def main():
    compass = pd.read_csv(HERE / "compass_pvals.csv")
    spvc    = pd.read_csv(HERE / "spvc_pvals.csv")
    n_rep   = (compass[compass.level == "omnibus"]
               .groupby(["scenario_id","expr","iso"])["rep"].nunique()
               .reset_index(name="n_rep"))
    print("ACTIVE truth (0-indexed t,k): " + ", ".join(f"{k}={sorted(v)}" for k, v in ACTIVE.items()))
    print(f"N_REP per scenario: {int(n_rep.n_rep.iloc[0])}")
    print(f"Tested cell types t∈{TESTED_T}, categories k∈{TESTED_K} (ref_ct=2, ref_cat=2)")
    l1 = build_l1(compass, spvc, n_rep)
    report_l1(l1)
    l2 = build_l2(compass, spvc, n_rep)
    report_l2(l2, compass, n_rep)
    l3 = build_l3(compass, spvc, n_rep)
    report_l3(l3, n_rep)
if __name__ == "__main__":
    main()
