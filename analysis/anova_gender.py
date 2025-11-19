# analysis/anova_gender.py

import numpy as np
import pandas as pd
from scipy import stats
from .descriptive_gender import get_numeric_df


def _has_variability(arr):
    arr = arr[~np.isnan(arr)]
    return len(np.unique(arr)) >= 2


def compute_anova_by_sex(df: pd.DataFrame, sex_col: str) -> dict:
    """
    ANOVA (one-way) comparing sex = 1 vs 2 for all numeric variables.
    """

    if sex_col not in df.columns:
        return {"error": f"No '{sex_col}' column found."}

    groups = df[sex_col].dropna().unique()
    if len(groups) != 2:
        return {"error": "Sex analysis requires exactly 2 groups."}

    g1, g2 = groups
    numeric = get_numeric_df(df)
    results = {}

    for var in numeric.columns:
        x = df[df[sex_col] == g1][var].dropna().values
        y = df[df[sex_col] == g2][var].dropna().values

        if len(x) < 2 or len(y) < 2:
            continue
        if not _has_variability(x) or not _has_variability(y):
            results[var] = {"F": None, "p": None, "reason": "Constant data"}
            continue

        try:
            F, p = stats.f_oneway(x, y)
            results[var] = {
                "F": float(F),
                "p": float(p),
                "group_sizes": {str(g1): len(x), str(g2): len(y)}
            }
        except Exception as e:
            results[var] = {"F": None, "p": None, "reason": str(e)}

    return results
