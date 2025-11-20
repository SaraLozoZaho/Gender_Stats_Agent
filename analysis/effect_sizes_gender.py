# analysis/effect_sizes_gender.py

import numpy as np
import pandas as pd

def cohen_d(x, y):
    """Compute Cohen's d for two independent groups."""
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return None

    pooled_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                         (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return None

    return (np.mean(x) - np.mean(y)) / pooled_sd


def compute_effect_sizes_numeric(df: pd.DataFrame, sex_col: str) -> dict:

    results = {}
    groups = df[sex_col].dropna().unique()
    if len(groups) != 2:
        return results

    g1, g2 = groups

    for col in df.columns:
        if df[col].dtype.kind not in "if":
            continue

        x = df[df[sex_col] == g1][col].dropna()
        y = df[df[sex_col] == g2][col].dropna()

        if len(x) < 2 or len(y) < 2:
            continue

        d = cohen_d(x.values, y.values)
        if d is not None:
            results[col] = {
                "cohen_d": float(d),
                "mean_diff": float(np.mean(x) - np.mean(y)),
            }

    return results
