# analysis/categorical_tests_gender.py

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

def compute_categorical_tests(df: pd.DataFrame, sex_col: str) -> dict:
    """
    Performs chi-square or Fisher exact test for all binary/categorical variables.
    Only variables with 2 categories are analyzed.
    """

    results = {}

    for col in df.columns:
        if col == sex_col:
            continue

        # Identify binary variables (0/1 or yes/no)
        uniques = df[col].dropna().unique()
        if len(uniques) != 2:
            continue

        # Build contingency table
        table = pd.crosstab(df[sex_col], df[col])

        if table.shape != (2, 2):
            continue

        # Try chi-square
        try:
            chi2, p, _, _ = chi2_contingency(table)
        except:
            chi2 = None
            p = None

        # Fisher exact
        try:
            oddsratio, pfisher = fisher_exact(table)
        except:
            oddsratio = None
            pfisher = None

        results[col] = {
            "chi2": float(chi2) if chi2 is not None else None,
            "p_chi2": float(p) if p is not None else None,
            "oddsratio": float(oddsratio) if oddsratio is not None else None,
            "p_fisher": float(pfisher) if pfisher is not None else None,
            "table": table.to_dict(),
        }

    return results
