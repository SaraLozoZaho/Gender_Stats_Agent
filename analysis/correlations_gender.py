# analysis/correlations_gender.py

import pandas as pd
import numpy as np
from .descriptive_gender import get_numeric_df


def compute_correlations_gender(df: pd.DataFrame) -> dict:
    numeric = get_numeric_df(df)
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
