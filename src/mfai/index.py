"""Stage 2: impedance-based access scores and the geometric-mean MFAI.

access_m = T_m / (T_m + d_m) with externally anchored reference distances
T_m, so scores are comparable across cities. MFAI is a weighted geometric
mean; MFAI-land drops the waterway mode, which is not available everywhere.
"""
import numpy as np
import pandas as pd

from . import paths
from .config import (COMPS, SHORT, LAND_MODES, T_REF, WEIGHTS_PRIMARY,
                     WEIGHTS_ALT, DIV_DIVERSITY_T as DIV_DIVERSITY_T_ANALYSIS)
from .io import apply_fallback_filter


def access_scores(df, T=None):
    T = T or T_REF
    a = pd.DataFrame(index=df.index)
    for raw, s in zip(COMPS, SHORT):
        a[s] = (T[s] / (T[s] + df[raw])
                if raw in df.columns and not df[raw].isna().all() else np.nan)
    return a


def weighted_geometric(acc, weights, modes=None):
    modes = modes or SHORT
    use = [m for m in modes if m in acc.columns and not acc[m].isna().all()]
    w = {m: weights[m] for m in use}
    tot = sum(w.values())
    w = {m: v/tot for m, v in w.items()}
    return np.exp(sum(w[m] * np.log(acc[m].clip(lower=1e-9)) for m in use)), w


def variance_decomposition(acc, w):
    """First-order sensitivity on the log scale, since aggregation is
    geometric (Paruolo, Saisana & Saltelli 2013). Nominal weight is not
    importance — report both."""
    use = list(w.keys())
    lv = {m: np.log(acc[m].clip(lower=1e-9)).var() for m in use}
    contrib = {m: (w[m]**2) * lv[m] for m in use}
    tot = sum(contrib.values())
    return pd.DataFrame({
        "weight":         [w[m] for m in use],
        "a_mean":         [acc[m].mean() for m in use],
        "a_median":       [acc[m].median() for m in use],
        "a_iqr":          [acc[m].quantile(.75)-acc[m].quantile(.25) for m in use],
        "variance_share": [contrib[m]/tot if tot > 0 else np.nan for m in use],
    }, index=use)


def add_mfai(city, weights=None, T=None):
    weights = weights or WEIGHTS_PRIMARY
    path = paths.nearest_csv(city)
    df = pd.read_csv(path)
    acc = access_scores(df, T)

    for s in SHORT:
        if s in acc.columns:
            df[f"access_{s}"] = acc[s].round(4)

    mfai, w = weighted_geometric(acc, weights)
    df["mfai"] = mfai.round(4)
    df["mfai_land"] = weighted_geometric(acc, weights, LAND_MODES)[0].round(4)
    for name, alt in WEIGHTS_ALT.items():
        df[f"mfai_{name}"] = weighted_geometric(acc, alt)[0].round(4)
    df.to_csv(path, index=False)

    kept = apply_fallback_filter(df)
    acck = access_scores(kept, T)
    print(f"\n[{city}] MFAI  n={len(df):,} unfiltered, {len(kept):,} at "
          f"T={DIV_DIVERSITY_T_ANALYSIS}")
    src = df["water_source"].iloc[0] if "water_source" in df else "network"
    if src != "network":
        print(f"    waterway via {src}")
    print(variance_decomposition(acck, w).round(4).to_string())
    for col, lbl in [("mfai", "mfai     "), ("mfai_land", "mfai_land")]:
        v = kept[col]
        print(f"    {lbl} mean {v.mean():.4f} | median {v.median():.4f} | "
              f"p10 {v.quantile(.10):.4f} | "
              f"IQR {v.quantile(.75)-v.quantile(.25):.4f}")
    return df
