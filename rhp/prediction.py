"""
Profile → behaviour prediction analysis (Block B, E8 + E9). CPU only.

The central RQ2 test: do the training-free profile scalars predict the
behavioural long-context scores across the panel? Three reads, deliberately
conservative for ~18 points (proposal §3.3, §5):

    (i)   single Spearman correlations, every profile metric × every target,
          Benjamini–Hochberg-corrected;                                    [E8]
    (ii)  a LOO-cross-validated, ≤3-predictor regression (guards against
          over-fitting 18 points);                                          [E8]
    (iii) family-demeaned correlation — family is a confound, so we also report
          the within-family relationship.                                   [E8]

E9 (test-retest) bounds all of the above: a metric cannot predict behaviour
better than it correlates with its own independent replication.

A weak / null result is a *finding*, reported with CIs, not a failure
(proposal §7).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.stats_utils import benjamini_hochberg

logger = logging.getLogger(__name__)

# Default profile predictors and behavioural targets (column names in the table).
# NOTE: freq_width is intentionally NOT a predictor — it is collinear with
# freq_com (same drop curve, identical Spearman on the shared non-NaN subset), so
# including both double-counts one signal. freq_width stays in the table for
# inspection but is excluded from correlations/regression.
PROFILE_PREDICTORS = ["n_heads", "frac", "gini", "zero_fraction", "layer_com",
                      "detector_jaccard", "freq_com",
                      "frequency_effect", "knockout_drop"]
NULL_FREQ_ABS = 0.05   # |frequency_effect| below this => no causal frequency
                       # dependence, so the spectral freq_com is noise (set NaN).
# M7 (separate experiment, results/utility/<model>.json) contributes two
# weight-space predictors when present.
UTILITY_PREDICTORS = ["utility_cohens_d", "utility_partial_spearman"]
ALL_PREDICTORS = PROFILE_PREDICTORS + UTILITY_PREDICTORS
# Long-context NIAH + the two discriminating RULER tasks carry the variance;
# plain niah_overall saturates at short context (pilot finding) so it is not the
# primary target. `niah_maxlen` (largest context with recall ≥ 0.5) is the most
# continuous long-context-ability target.
BEHAVIOR_TARGETS = ["niah_maxlen", "niah_long", "ruler_multivalue", "ruler_vartrack"]
# Targets for which we also report the within-family (family-demeaned)
# correlation. niah_maxlen is the primary categorical long-context proxy;
# niah_long and ruler_vartrack are the continuous targets that carry the RQ2
# confound diagnosis (the raw correlation is confounded by context-window class,
# so the within-family signal is the honest read). Each target is BH-corrected
# across the predictor set independently.
FAMILY_DEMEAN_TARGETS = ["niah_maxlen", "niah_long", "ruler_vartrack"]


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------

def build_table(results: list[dict]) -> pd.DataFrame:
    """
    Flatten per-model results into one row per model.

    Each ``result`` is expected to carry ``model``, ``family``, a
    ``profile.scalars`` dict (from ``rhp.profile.build_profile``) and a
    ``behavior`` dict with ``niah_overall``/``niah_worst_pos`` and a
    ``ruler.task_means`` mapping. Missing fields become NaN.
    """
    rows = []
    for r in results:
        scalars = (r.get("profile", {}).get("scalars", {}) or {})
        beh = r.get("behavior", {})
        ruler_means = (beh.get("ruler", {}) or {}).get("task_means", {})
        row = {"model": r.get("model"), "family": r.get("family", "unknown")}
        for k in PROFILE_PREDICTORS:
            row[k] = scalars.get(k, np.nan)
        row["freq_width"] = scalars.get("freq_width", np.nan)   # table only, not a predictor
        # Null-frequency gate: a model whose causal frequency effect is ~0 has no
        # real spectral structure, so its freq_com centre-of-mass is noise (e.g.
        # Llama: dose effect 0, but a non-zero freq_com from a near-flat curve).
        # Drop freq_com for such models so it cannot pollute the correlation.
        fe = scalars.get("frequency_effect", np.nan)
        if not (abs(fe) >= NULL_FREQ_ABS):   # NaN or below threshold
            row["freq_com"] = np.nan
            row["freq_width"] = np.nan
        # M7 utility predictors come from the separate utility/<model>.json,
        # merged onto the result under "utility" by the collector.
        util = r.get("utility", {}) or {}
        row["utility_cohens_d"] = util.get("cohens_d", np.nan)
        row["utility_partial_spearman"] = util.get("partial_spearman", np.nan)
        row["niah_overall"] = beh.get("niah_overall", np.nan)
        row["niah_long"] = beh.get("niah_long", beh.get("niah_overall", np.nan))
        # niah_maxlen: use stored value, else backfill from per-context recall
        # (keeps results produced before this field was added usable).
        maxlen = beh.get("niah_maxlen")
        if maxlen is None:
            pc = beh.get("niah_per_context", {}) or {}
            rel = [int(c) for c, v in pc.items() if v is not None and v == v and v >= 0.5]
            maxlen = float(max(rel)) if rel else np.nan
        row["niah_maxlen"] = maxlen
        row["niah_worst_pos"] = beh.get("niah_worst_pos", np.nan)
        # per-task RULER (the discriminating targets) + the mean
        row["ruler_multikey"] = ruler_means.get("multikey", np.nan)
        row["ruler_multivalue"] = ruler_means.get("multivalue", np.nan)
        row["ruler_vartrack"] = ruler_means.get("vartrack", np.nan)
        row["ruler_mean"] = float(np.mean(list(ruler_means.values()))) if ruler_means else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E8 (i) — single correlations with BH-FDR
# ---------------------------------------------------------------------------

def single_correlations(
    df: pd.DataFrame,
    predictors: list[str] | None = None,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Spearman ρ for every (predictor, target), with BH-corrected p-values."""
    predictors = predictors or [p for p in ALL_PREDICTORS if p in df.columns]
    targets = targets or [t for t in BEHAVIOR_TARGETS if t in df.columns]
    recs, pvals = [], []
    for p in predictors:
        for t in targets:
            sub = df[[p, t]].dropna()
            if len(sub) < 4 or sub[p].nunique() < 2 or sub[t].nunique() < 2:
                rho, pv = np.nan, np.nan
            else:
                rho, pv = stats.spearmanr(sub[p], sub[t])
            recs.append({"predictor": p, "target": t, "spearman_rho": float(rho),
                         "p_value": float(pv), "n": len(sub)})
            pvals.append(pv)
    bh = benjamini_hochberg(pvals)
    out = pd.DataFrame(recs)
    out["p_adjusted_bh"] = bh["p_adjusted"]
    out["significant_bh"] = bh["rejected"]
    return out.sort_values("p_value", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# E8 (ii) — LOO-cross-validated constrained regression
# ---------------------------------------------------------------------------

def loo_regression(df: pd.DataFrame, predictors: list[str], target: str) -> dict:
    """
    Leave-one-out CV for a ≤3-predictor linear regression (E8).

    Returns the out-of-sample R² (and Spearman of predicted vs actual), the
    honest accuracy of "use the profile to predict the benchmark". Over-fitting
    18 points is the risk; LOO is the cheapest guard.
    """
    from sklearn.linear_model import LinearRegression

    if len(predictors) > 3:
        raise ValueError("Keep to ≤3 predictors for 18 points (proposal §3.3).")
    sub = df[predictors + [target]].dropna()
    n = len(sub)
    if n < 5:
        return {"n": n, "loo_r2": float("nan"), "loo_spearman": float("nan"),
                "predictors": predictors, "target": target,
                "note": "too few complete cases"}
    X = sub[predictors].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    # standardise predictors (fit on the training fold each iteration)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-9
        model = LinearRegression().fit((X[mask] - mu) / sd, y[mask])
        preds[i] = model.predict(((X[i] - mu) / sd).reshape(1, -1))[0]
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sp = float(stats.spearmanr(y, preds)[0]) if np.std(preds) > 0 else float("nan")
    return {"n": n, "loo_r2": r2, "loo_spearman": sp,
            "predictors": predictors, "target": target}


# ---------------------------------------------------------------------------
# E8 (iii) — family-demeaned (within-family) correlation
# ---------------------------------------------------------------------------

def family_demeaned_correlation(df: pd.DataFrame, predictor: str, target: str) -> dict:
    """Spearman after subtracting each family's mean from predictor and target."""
    sub = df[[predictor, target, "family"]].dropna()
    if len(sub) < 4:
        return {"predictor": predictor, "target": target, "n": len(sub),
                "within_family_spearman": float("nan"), "p_value": float("nan")}
    dm = sub.copy()
    for col in (predictor, target):
        dm[col] = dm.groupby("family")[col].transform(lambda x: x - x.mean())
    if dm[predictor].nunique() < 2 or dm[target].nunique() < 2:
        rho, pv = float("nan"), float("nan")
    else:
        rho, pv = stats.spearmanr(dm[predictor], dm[target])
    return {"predictor": predictor, "target": target, "n": len(sub),
            "within_family_spearman": float(rho), "p_value": float(pv),
            "n_families": int(sub["family"].nunique())}


# ---------------------------------------------------------------------------
# E9 — profile test-retest
# ---------------------------------------------------------------------------

def within_model_reliability(rep_a: dict, rep_b: dict) -> dict:
    """
    Head-set reliability ceiling for ONE model from two independent runs (E9).

    ``rep_a``/``rep_b`` are two profile results for the **same** model at two
    seeds / sample sets. Returns the head-set Jaccard (both detectors) and the
    dense-score Spearman — the *ceiling* against which an inheritance Jaccard
    must be read: a base→child Jaccard equal to this reliability means "as
    similar as the model is to itself" (full preservation), not "changed". This
    is the denominator the pre-registration uses instead of an absolute 1.0.
    """
    from src.stats_utils import jaccard
    from scipy import stats

    def _heads(r, key):
        return [tuple(h) for h in r.get(key, [])]

    def _score_spearman(a, b):
        a, b = np.asarray(a, dtype=float).flatten(), np.asarray(b, dtype=float).flatten()
        if a.shape != b.shape or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            return float("nan")
        return float(stats.spearmanr(a, b)[0])

    ja = jaccard(_heads(rep_a, "argmax_heads"), _heads(rep_b, "argmax_heads"))
    jc = jaccard(_heads(rep_a, "copy_heads"), _heads(rep_b, "copy_heads"))
    return {
        "model": rep_a.get("model"),
        "argmax_jaccard": ja["jaccard"],
        "copy_jaccard": jc["jaccard"],
        "argmax_score_spearman": _score_spearman(
            rep_a.get("argmax_scores"), rep_b.get("argmax_scores")),
        "copy_score_spearman": _score_spearman(
            rep_a.get("copy_scores"), rep_b.get("copy_scores")),
        "seed_a": rep_a.get("seed"), "seed_b": rep_b.get("seed"),
    }


def test_retest(rep1: list[dict], rep2: list[dict], metrics: list[str] | None = None) -> pd.DataFrame:
    """
    Per-metric test-retest correlation between two independent profile runs (E9).

    ``rep1``/``rep2`` are two lists of per-model results (same models, two
    independent sample sets). The returned reliability ceilings are reported as
    upper bounds in E8.
    """
    metrics = metrics or PROFILE_PREDICTORS
    t1 = build_table(rep1).set_index("model")
    t2 = build_table(rep2).set_index("model")
    common = t1.index.intersection(t2.index)
    recs = []
    for m in metrics:
        if m not in t1.columns or m not in t2.columns:
            continue
        a = t1.loc[common, m].to_numpy(dtype=float)
        b = t2.loc[common, m].to_numpy(dtype=float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 4 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
            rho = np.nan
        else:
            rho = float(stats.spearmanr(a[ok], b[ok])[0])
        recs.append({"metric": m, "test_retest_spearman": rho, "n": int(ok.sum())})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_prediction_analysis(
    results: list[dict],
    predictors: list[str] | None = None,
    targets: list[str] | None = None,
) -> dict:
    """Run E8 (i)+(iii) and a default 3-predictor LOO regression per target."""
    df = build_table(results)
    predictors = predictors or [p for p in ALL_PREDICTORS if p in df.columns]
    targets = targets or [t for t in BEHAVIOR_TARGETS if t in df.columns]

    corr = single_correlations(df, predictors, targets)
    # pick the 3 predictors with strongest |rho| against the primary target
    primary = targets[0]
    ranked = (corr[corr.target == primary]
              .assign(absr=lambda d: d.spearman_rho.abs())
              .sort_values("absr", ascending=False))
    top3 = [p for p in ranked.predictor.tolist() if p in predictors][:3]
    loo = {t: loo_regression(df, top3, t) for t in targets} if top3 else {}
    # Family-demeaned correlations, WITH BH correction across the predictor set
    # (was previously raw p only — a reviewer would reject "significant" on
    # uncorrected p). Computed for every FAMILY_DEMEAN_TARGETS entry present, not
    # just the primary, so the RQ2 confound diagnosis (niah_long, ruler_vartrack)
    # is backed by stored numbers rather than a live figure re-computation. BH is
    # applied within each target's predictor family independently.
    demean_targets = [t for t in FAMILY_DEMEAN_TARGETS if t in df.columns]
    if primary not in demean_targets and primary in df.columns:
        demean_targets = [primary] + demean_targets
    fam = []
    for tgt in demean_targets:
        fam_t = [family_demeaned_correlation(df, p, tgt) for p in predictors]
        bh_t = benjamini_hochberg([f.get("p_value", float("nan")) for f in fam_t])
        for f, padj, rej in zip(fam_t, bh_t["p_adjusted"], bh_t["rejected"]):
            f["p_adjusted_bh"] = padj
            f["significant_bh"] = bool(rej)
        fam.extend(fam_t)

    return {
        "n_models": len(df),
        "table": df.to_dict(orient="records"),
        "single_correlations": corr.to_dict(orient="records"),
        "loo_top3_predictors": top3,
        "loo_regression": loo,
        "family_demeaned": fam,
    }
