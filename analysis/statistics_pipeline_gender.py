# analysis/statistics_pipeline_gender.py

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .descriptive_gender import compute_descriptive_stats_gender
from .correlations_gender import compute_correlations_gender
from .anova_gender import compute_anova_by_sex


# ======================================================
# HELPER: bootstrap CI
# ======================================================

def bootstrap_ci(x, y, n=2000):
    """Compute 95% bootstrap CI for mean difference."""
    x = np.array(x)
    y = np.array(y)
    boots = []
    for _ in range(n):
        bx = np.random.choice(x, len(x), replace=True)
        by = np.random.choice(y, len(y), replace=True)
        boots.append(bx.mean() - by.mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ======================================================
# T-TEST / MANN–WHITNEY / COHEN’S d
# ======================================================

def compute_group_tests(df, sex_col="Sex"):
    if sex_col not in df.columns:
        return {}

    sex_vals = df[sex_col].dropna().unique()
    if len(sex_vals) != 2:
        return {}

    g1, g2 = sex_vals
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}

    for var in numeric_cols:
        x = df[df[sex_col] == g1][var].dropna().values
        y = df[df[sex_col] == g2][var].dropna().values

        if len(x) < 3 or len(y) < 3:
            continue

        # Normality
        try:
            p1 = stats.shapiro(x)[1]
            p2 = stats.shapiro(y)[1]
            normal = p1 > 0.05 and p2 > 0.05
        except Exception:
            normal = False

        # Test selection
        if normal:
            t, p = stats.ttest_ind(x, y, equal_var=False)
            test_name = "t-test"
            effect = float(x.mean() - y.mean())
            cohen_d = float(effect / np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2)/2))
        else:
            t, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            test_name = "Mann–Whitney U"
            effect = float(np.median(x) - np.median(y))
            cohen_d = None  # not meaningful for nonparametric

        # Bootstrap CI
        try:
            ci_low, ci_high = bootstrap_ci(x, y)
        except Exception:
            ci_low = ci_high = None

        results[var] = {
            "test": test_name,
            "p": float(p),
            "effect": effect,
            "cohen_d": cohen_d,
            "ci": [ci_low, ci_high],
            "group_sizes": {str(g1): len(x), str(g2): len(y)},
        }

    return results


# ======================================================
# χ² TEST FOR CATEGORICAL VARIABLES
# ======================================================

def compute_categorical_tests(df, sex_col="Sex"):
    if sex_col not in df.columns:
        return {}

    categorical_cols = [
        c for c in df.columns
        if df[c].dtype == "object" or df[c].dtype.name.startswith("category")
    ]

    results = {}

    for col in categorical_cols:
        try:
            table = pd.crosstab(df[sex_col], df[col])
            if table.shape[1] < 2:
                continue
            chisq, p, dof, exp = stats.chi2_contingency(table)
            results[col] = {
                "chi2": float(chisq),
                "p": float(p),
                "table": table.to_dict(),
            }
        except Exception:
            continue

    return results


# ======================================================
# Logistic regression
# ======================================================

def compute_logistic_regression(df, sex_col="Sex"):
    """
    Tries to find a binary outcome automatically.
    """
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]

    if not binary_cols:
        return {}

    ycol = binary_cols[0]

    if sex_col not in df.columns:
        return {}

    df2 = df[[ycol, sex_col]].dropna()

    # encode
    df2["sex_encoded"] = df2[sex_col].astype("category").cat.codes

    X = sm.add_constant(df2["sex_encoded"])
    y = df2[ycol]

    try:
        model = sm.Logit(y, X).fit(disp=False)
        OR = float(np.exp(model.params["sex_encoded"]))
        p = float(model.pvalues["sex_encoded"])
        return {
            "outcome": ycol,
            "OR": OR,
            "p": p,
            "n": len(df2),
        }
    except Exception:
        return {}


# ======================================================
# MAIN PIPELINE
# ======================================================

def build_gender_stats_payload(df: pd.DataFrame, sex_col="Sex"):
    payload = {}

    # Dataset info
    payload["dataset_info"] = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "count_by_sex": df[sex_col].value_counts(dropna=True).to_dict()
            if sex_col in df.columns else {}
    }

    # Blocks
    payload["descriptive"] = compute_descriptive_stats_gender(df, sex_col)
    payload["correlations"] = compute_correlations_gender(df)
    payload["anova_sex"] = compute_anova_by_sex(df, sex_col)
    payload["t_tests"] = compute_group_tests(df, sex_col)
    payload["categorical_tests"] = compute_categorical_tests(df, sex_col)
    payload["logistic_regression"] = compute_logistic_regression(df, sex_col)

    # Summary indicator
    payload["gender_gap_indicators"] = {
        "significant_t_tests": len([v for v in payload["t_tests"].values() if v["p"] < 0.05]),
        "significant_anova": len([
            v for v in payload["anova_sex"].values()
            if isinstance(v, dict) and v.get("p") is not None and v["p"] < 0.05
        ])
    }

    return payload
