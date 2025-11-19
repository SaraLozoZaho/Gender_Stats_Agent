# analysis/statistics_pipeline_gender.py

import pandas as pd
import numpy as np
from scipy import stats


# =========================================================
# HELPERS
# =========================================================

def _safe_mean(series: pd.Series):
    s = series.dropna()
    return float(s.mean()) if len(s) > 0 else None

def _has_variability(arr: np.ndarray):
    arr = arr[~np.isnan(arr)]
    return len(np.unique(arr)) >= 2

def _numeric_df(df):
    return df.select_dtypes(include=[np.number])


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

def compute_basic_descriptives(df: pd.DataFrame, sex_col: str) -> dict:
    numeric = _numeric_df(df)
    overall = numeric.describe().T.round(4).to_dict(orient="index")

    by_sex = {}
    if sex_col in df.columns:
        for sex, sub in df.groupby(sex_col):
            if pd.isna(sex):
                continue
            desc = sub[numeric.columns].describe().T.round(4).to_dict(orient="index")
            by_sex[str(sex)] = desc

    return {
        "overall": overall,
        "by_sex": by_sex,
        "numeric_columns": list(numeric.columns)
    }


# =========================================================
# CORRELATIONS
# =========================================================

def compute_correlations(df: pd.DataFrame) -> dict:
    numeric = _numeric_df(df)
    corr = numeric.corr().round(3)

    strong = []
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) >= 0.7:
                strong.append({"var1": c1, "var2": c2, "r": float(r)})

    return {
        "matrix": corr.to_dict(),
        "strong_pairs": strong
    }


# =========================================================
# ANOVA BETWEEN SEX GROUPS (1 vs 2)
# =========================================================

def compute_anova_sex(df: pd.DataFrame, sex_col: str) -> dict:
    if sex_col not in df.columns:
        return {"error": f"No '{sex_col}' column found"}

    numeric = _numeric_df(df)
    result = {}

    groups = df[sex_col].dropna().unique()
    if len(groups) != 2:
        return {"error": "Sex analysis requires exactly 2 groups (e.g., 1 = male, 2 = female)."}

    g1, g2 = groups

    for var in numeric.columns:
        x = df[df[sex_col] == g1][var].dropna().values
        y = df[df[sex_col] == g2][var].dropna().values

        if len(x) < 2 or len(y) < 2:
            continue

        if not _has_variability(x) or not _has_variability(y):
            reason = "Constant values, cannot compute ANOVA"
            result[var] = {"F": None, "p": None, "reason": reason}
            continue

        try:
            F, p = stats.f_oneway(x, y)
            result[var] = {
                "F": float(F),
                "p": float(p),
                "groups": {str(g1): len(x), str(g2): len(y)}
            }
        except Exception as e:
            result[var] = {"F": None, "p": None, "reason": str(e)}

    return result


# =========================================================
# SIMPLE REGRESSIONS
# =========================================================

def compute_simple_regressions(df: pd.DataFrame) -> dict:
    from scipy.stats import linregress

    result = {}
    numeric = _numeric_df(df)
    cols = list(numeric.columns)

    for i, x in enumerate(cols):
        for y in cols[i+1:]:
            sub = df[[x, y]].dropna()
            if len(sub) < 4:
                continue

            if not _has_variability(sub[x].values) or not _has_variability(sub[y].values):
                continue

            try:
                r = linregress(sub[x], sub[y])
                result[f"{y}_vs_{x}"] = {
                    "x": x,
                    "y": y,
                    "coef": float(r.slope),
                    "intercept": float(r.intercept),
                    "r2": float(r.rvalue**2),
                    "p": float(r.pvalue),
                    "stderr": float(r.stderr),
                    "n": len(sub),
                }
            except Exception:
                continue

    return result


# =========================================================
# FINAL PAYLOAD (like build_stats_payload)
# =========================================================

def build_gender_stats_payload(df: pd.DataFrame, sex_col: str = "Sex") -> dict:

    numeric = _numeric_df(df)

    payload = {
        "dataset_info": {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns),
            "count_by_sex": df[sex_col].value_counts(dropna=True).to_dict()
                if sex_col in df.columns else {},
        },

        "descriptive": compute_basic_descriptives(df, sex_col),
        "correlations": compute_correlations(df),
        "anova_sex": compute_anova_sex(df, sex_col),
        "regressions": compute_simple_regressions(df),
    }

    # Simple gender-gap indicator
    try:
        sig = [v for v in payload["anova_sex"].values() if isinstance(v, dict) and v.get("p") and v["p"] < 0.05]
        payload["gender_gap_indicators"] = {
            "significant_variables": len(sig)
        }
    except Exception:
        payload["gender_gap_indicators"] = {}

    return payload
