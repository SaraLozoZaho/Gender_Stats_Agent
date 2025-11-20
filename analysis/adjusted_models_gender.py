# analysis/adjusted_models_gender.py

import pandas as pd
import statsmodels.api as sm
import numpy as np

def logistic_adjusted(df, outcome, sex_col, adjust_cols):
    """Logistic regression: outcome ~ sex + adjust_cols."""
    cols = [sex_col] + adjust_cols

    sub = df[[outcome] + cols].dropna()
    if sub[outcome].nunique() != 2:
        return None

    # encode sex
    sub["sex_bin"] = sub[sex_col].astype("category").cat.codes

    X = sm.add_constant(sub[["sex_bin"] + adjust_cols])
    y = sub[outcome]

    try:
        m = sm.Logit(y, X).fit(disp=False)
        OR = np.exp(m.params["sex_bin"])
        CI_low = np.exp(m.conf_int().loc["sex_bin"][0])
        CI_high = np.exp(m.conf_int().loc["sex_bin"][1])
        return {
            "model": "logistic",
            "outcome": outcome,
            "OR_sex": float(OR),
            "CI": [float(CI_low), float(CI_high)],
            "p": float(m.pvalues["sex_bin"]),
            "n": int(len(sub))
        }
    except:
        return None


def linear_adjusted(df, outcome, sex_col, adjust_cols):
    """Linear regression: outcome ~ sex + adjust_cols."""
    cols = [sex_col] + adjust_cols
    sub = df[[outcome] + cols].dropna()

    if len(sub) < 10:
        return None

    # encode sex
    sub["sex_bin"] = sub[sex_col].astype("category").cat.codes

    X = sm.add_constant(sub[["sex_bin"] + adjust_cols])
    y = sub[outcome]

    try:
        m = sm.OLS(y, X).fit()
        beta = m.params["sex_bin"]
        CI_low, CI_high = m.conf_int().loc["sex_bin"]
        return {
            "model": "linear",
            "outcome": outcome,
            "beta_sex": float(beta),
            "CI": [float(CI_low), float(CI_high)],
            "p": float(m.pvalues["sex_bin"]),
            "n": int(len(sub))
        }
    except:
        return None
