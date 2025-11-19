# GENDER_STATS_AGENT/agent/gender_agent_core.py

import numpy as np
import pandas as pd
from scipy import stats

from utils.sex_gender_utils import detect_sex_gender_vars


# ========================================================================
# HELPER: select numeric columns
# ========================================================================

def _numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ========================================================================
# 1. DESCRIPTIVES
# ========================================================================

def compute_descriptives(df, sex_gender_info):
    result = {
        "by_sex": {},
        "by_gender": {}
    }

    # --- SEX ---
    if sex_gender_info.sex_col:
        col = sex_gender_info.sex_col
        groups = df[col].dropna().unique()

        for g in groups:
            sub = df[df[col] == g]
            desc = {}
            for var in _numeric_columns(df):
                x = sub[var].dropna()
                if len(x) > 0:
                    desc[var] = {
                        "mean": float(np.mean(x)),
                        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None
                    }
            result["by_sex"][str(g)] = desc

    # --- GENDER ---
    if sex_gender_info.gender_col:
        col = sex_gender_info.gender_col
        groups = df[col].dropna().unique()

        for g in groups:
            sub = df[df[col] == g]
            desc = {}
            for var in _numeric_columns(df):
                x = sub[var].dropna()
                if len(x) > 0:
                    desc[var] = {
                        "mean": float(np.mean(x)),
                        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None
                    }
            result["by_gender"][str(g)] = desc

    return result


# ========================================================================
# 2. GROUP TESTS (t-test / Mann–Whitney)
# ========================================================================

def compute_group_tests(df, sex_gender_info):
    result = {}

    # Prefer SEX over GENDER if both exist
    group_col = sex_gender_info.sex_col or sex_gender_info.gender_col
    if not group_col:
        return {}

    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        # tests only meaningful for 2 groups
        return {}

    g1, g2 = groups
    numeric_cols = _numeric_columns(df)

    for var in numeric_cols:
        x = df[df[group_col] == g1][var].dropna()
        y = df[df[group_col] == g2][var].dropna()

        if len(x) < 3 or len(y) < 3:
            continue

        # Normality check
        normal = False
        try:
            _, p1 = stats.shapiro(x)
            _, p2 = stats.shapiro(y)
            normal = (p1 > 0.05 and p2 > 0.05)
        except Exception:
            pass

        if normal:
            test_name = "t-test"
            t, p = stats.ttest_ind(x, y, equal_var=False)
            effect = float(np.mean(x) - np.mean(y))
        else:
            test_name = "Mann-Whitney U"
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            effect = float(np.median(x) - np.median(y))

        # Confidence interval for effect (bootstrap 2000 samples)
        try:
            boot = []
            for _ in range(1500):
                bx = np.random.choice(x, len(x), replace=True)
                by = np.random.choice(y, len(y), replace=True)
                boot.append(np.mean(bx) - np.mean(by))
            ci_low = float(np.percentile(boot, 2.5))
            ci_high = float(np.percentile(boot, 97.5))
            ci = [ci_low, ci_high]
        except Exception:
            ci = None

        result[var] = {
            "test_name": test_name,
            "p": float(p),
            "effect": effect,
            "ci": ci
        }

    return result


# ========================================================================
# 3. SIMPLE PREDICTIVE MODELS
# ========================================================================

def compute_models(df, sex_gender_info):
    """
    Very simple model:
    - If binary outcome exists (column name contains 'outcome'), run logistic regression.
    - Sex/Gender is included as main predictor.
    """
    import statsmodels.api as sm

    result = {}

    # Find binary outcome
    candidate_cols = [c for c in df.columns if "outcome" in c.lower()]
    if not candidate_cols:
        return result

    outcome_col = candidate_cols[0]
    y = df[outcome_col]

    # Predictor is SEX or GENDER
    pred = sex_gender_info.sex_col or sex_gender_info.gender_col
    if not pred:
        return result

    # Encode categorical as 0/1
    df2 = df[[pred, outcome_col]].dropna()
    if df2[pred].nunique() != 2:
        return result

    df2["pred_encoded"] = df2[pred].astype("category").cat.codes
    X = sm.add_constant(df2["pred_encoded"])
    y = df2[outcome_col]

    try:
        model = sm.Logit(y, X).fit(disp=False)
        OR = float(np.exp(model.params["pred_encoded"]))
        p = float(model.pvalues["pred_encoded"])
        result["logistic_sexgender"] = {
            "metric": f"Odds Ratio = {OR:.3f}, p={p:.3e}",
            "sex_effect": OR,
            "notes": None
        }
    except Exception:
        pass

    return result


# ========================================================================
# 4. MAIN PIPELINE: build_gender_stats_payload
# ========================================================================

def build_gender_stats_payload(df: pd.DataFrame) -> dict:

    # Detect sex/gender variables
    sex_gender_info = detect_sex_gender_vars(df)

    # Basic dataset info
    stats = {
        "dataset_info": {
            "n_total": len(df),
            "count_by_sex": {},
            "count_by_gender": {}
        },
        "descriptive": {},
        "group_tests": {},
        "models": {},
        "gender_gap_indicators": {},
        "sex_gender_info": {
            "sex_col": sex_gender_info.sex_col,
            "gender_col": sex_gender_info.gender_col
        }
    }

    # Counts
    if sex_gender_info.sex_col:
        stats["dataset_info"]["count_by_sex"] = (
            df[sex_gender_info.sex_col].value_counts(dropna=True).to_dict()
        )

    if sex_gender_info.gender_col:
        stats["dataset_info"]["count_by_gender"] = (
            df[sex_gender_info.gender_col].value_counts(dropna=True).to_dict()
        )

    # Descriptives
    stats["descriptive"] = compute_descriptives(df, sex_gender_info)

    # Group tests
    stats["group_tests"] = compute_group_tests(df, sex_gender_info)

    # Models
    stats["models"] = compute_models(df, sex_gender_info)

    # Simple gap indicators (interpretation-level)
    try:
        sig = [v for v in stats["group_tests"].values() if v.get("p") < 0.05]
        stats["gender_gap_indicators"]["significant_differences"] = len(sig)
    except Exception:
        pass

    return stats
