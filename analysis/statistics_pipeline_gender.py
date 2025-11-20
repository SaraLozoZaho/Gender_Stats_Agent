# analysis/statistics_pipeline_gender.py

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .descriptive_gender import compute_descriptive_stats_gender
from .correlations_gender import compute_correlations_gender
from .anova_gender import compute_anova_by_sex


# ======================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ======================================================

def bootstrap_ci(x, y, n=2000):
    """Safe 95% CI for mean difference."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None, None

    boots = []
    for _ in range(n):
        bx = np.random.choice(x, len(x), replace=True)
        by = np.random.choice(y, len(y), replace=True)
        boots.append(bx.mean() - by.mean())

    return (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


# ======================================================
# GROUP TESTS (t-test / Mann–Whitney)
# ======================================================

def compute_group_tests(df, sex_col="Sex"):
    """Performs t-test or Mann–Whitney U and effect sizes."""
    if sex_col not in df.columns:
        return {}

    sex_vals = np.sort(df[sex_col].dropna().unique())
    if len(sex_vals) != 2:
        return {}

    g1, g2 = sex_vals

    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}

    for var in numeric_vars:
        x = df[df[sex_col] == g1][var].dropna().to_numpy()
        y = df[df[sex_col] == g2][var].dropna().to_numpy()

        if len(x) < 3 or len(y) < 3:
            continue

        # Normality check
        try:
            p1 = stats.shapiro(x)[1]
            p2 = stats.shapiro(y)[1]
            normal = (p1 > 0.05 and p2 > 0.05)
        except Exception:
            normal = False

        if normal:
            stat, p = stats.ttest_ind(x, y, equal_var=False)
            effect = float(x.mean() - y.mean())

            pooled_sd = np.sqrt((x.std(ddof=1)**2 + y.std(ddof=1)**2) / 2)
            cohen_d = float(effect / pooled_sd) if pooled_sd > 0 else None

            test_name = "t-test"
        else:
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            effect = float(np.median(x) - np.median(y))
            cohen_d = None
            test_name = "Mann–Whitney U"

        ci_low, ci_high = bootstrap_ci(x, y)

        results[var] = {
            "test": test_name,
            "p": float(p),
            "effect": float(effect),
            "cohen_d": cohen_d,
            "ci": [ci_low, ci_high],
            "group_sizes": {str(g1): len(x), str(g2): len(y)},
        }

    return results


# ======================================================
# CHI-SQUARE TESTS
# ======================================================

def compute_categorical_tests(df, sex_col="Sex"):
    """χ² tests for all categorical variables."""
    if sex_col not in df.columns:
        return {}

    results = {}
    sex_vals = df[sex_col].dropna().unique()

    for col in df.columns:
        if col == sex_col:
            continue

        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            try:
                table = pd.crosstab(df[sex_col], df[col])
                if table.shape[1] < 2:
                    continue
                chi2, p, dof, exp = stats.chi2_contingency(table)
                results[col] = {
                    "chi2": float(chi2),
                    "p": float(p),
                    "table": table.to_dict(),
                }
            except Exception:
                continue

    return results


# ======================================================
# LOGISTIC REGRESSION (FINAL STABLE VERSION)
# ======================================================

def compute_logistic_regression(df, sex_col="Sex"):
    """
    Automatically detects a binary (0/1) outcome and fits:
        outcome ~ Sex
    Fully robust:
    - No .cat misuse
    - No DataFrame/Series confusion
    - No length mismatch
    - No crashes
    """

    # 1. Identify binary 0/1 outcomes
    binary_candidates = []
    for col in df.columns:
        non_missing = df[col].dropna()
        if len(non_missing) > 3 and non_missing.isin([0, 1]).all():
            binary_candidates.append(col)

    if not binary_candidates:
        return {}

    ycol = binary_candidates[0]

    if sex_col not in df.columns:
        return {}

    # 2. Clean subset
    df2 = df[[ycol, sex_col]].dropna().copy()

    # Extra cleanup inside df2 (important if Excel had hidden duplicate headers)
    df2.columns = (
        df2.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    df2 = df2.loc[:, ~df2.columns.duplicated()]

    # 3. Sex must be a Series, not DataFrame
    sex_raw = df2[sex_col].squeeze()   # ALWAYS a Series
    sex_sorted = sorted(pd.unique(sex_raw))   # more robust than Series.unique()

    # 4. Encode sex values as integers (0/1)
    mapping = {v: i for i, v in enumerate(sex_sorted)}   # Example: {0:0, 1:1}
    df2["sex_encoded"] = sex_raw.map(mapping).astype(int)

    # 5. Build logistic regression model
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
            "n": int(len(df2)),
            "sex_mapping": mapping,
            "model_summary": str(model.summary()),
        }

    except Exception as e:
        # Always return something instead of crashing
        return {"error": str(e)}



# ======================================================
# MAIN PIPELINE
# ======================================================
def build_gender_stats_payload(df: pd.DataFrame, sex_col="Sex"):
    """Builds the full JSON package used by the GUI and expert agent."""
    payload = {}

    payload["dataset_info"] = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "count_by_sex": df[sex_col].value_counts(dropna=True).to_dict()
            if sex_col in df.columns else {},
    }

    payload["descriptive"] = compute_descriptive_stats_gender(df, sex_col)
    payload["correlations"] = compute_correlations_gender(df)
    payload["anova_sex"] = compute_anova_by_sex(df, sex_col)
    payload["t_tests"] = compute_group_tests(df, sex_col)
    payload["categorical_tests"] = compute_categorical_tests(df, sex_col)
    payload["logistic_regression"] = compute_logistic_regression(df, sex_col)

    payload["gender_gap_indicators"] = {
        "significant_t_tests": len([
            v for v in payload["t_tests"].values()
            if v["p"] < 0.05
        ]),
        "significant_anova": len([
            v for v in payload["anova_sex"].values()
            if isinstance(v, dict) and v.get("p") is not None and v["p"] < 0.05
        ]),
        "significant_categorical": len([
            v for v in payload["categorical_tests"].values()
            if v["p"] < 0.05
        ])
    }

    return payload
