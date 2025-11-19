# analysis/regression_gender.py

import numpy as np
import pandas as pd
from scipy.stats import linregress
from .descriptive_gender import get_numeric_df


def compute_regressions_gender(df: pd.DataFrame) -> dict:
    """
    All pairwise simple regressions among numeric columns
    (just like respiratory agent).
    """
    numeric = get_numeric_df(df)
    cols = list(numeric.columns)
    result = {}

    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            sub = df[[x, y]].dropna()
            if len(sub) < 4:
                continue

            try:
                r = linregress(sub[x], sub[y])
                key = f"{y}_vs_{x}"
                result[key] = {
                    "x": x,
                    "y": y,
                    "coef": float(r.slope),
                    "intercept": float(r.intercept),
                    "r2": float(r.rvalue ** 2),
                    "p": float(r.pvalue),
                    "stderr": float(r.stderr),
                    "n": len(sub),
                }
            except Exception:
                continue

    return result
