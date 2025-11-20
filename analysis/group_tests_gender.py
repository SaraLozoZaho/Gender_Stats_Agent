# analysis/group_tests_gender.py

import numpy as np
import pandas as pd
from scipy import stats
from .descriptive_gender import get_numeric_df


def _is_binary(series: pd.Series) -> bool:
    """Detect 0/1 tipo binario."""
    vals = series.dropna().unique()
    if len(vals) < 2:
        return False
    return set(vals).issubset({0, 1})


def compute_group_tests_gender(df: pd.DataFrame, sex_col: str = "Sex") -> dict:
    """
    Para cada variable numérica:
    - Si es 0/1 -> χ² (o Fisher si hace falta)
    - Si es continua -> Welch t-test (y si falla, Mann–Whitney)

    Devuelve:
    { var: {test_name, type, p, effect, ...}, ... }
    """
    if sex_col not in df.columns:
        return {}

    # Grupos de sexo (0/1 o 1/2, da igual)
    sexes = df[sex_col].dropna().unique()
    if len(sexes) != 2:
        return {}

    g1, g2 = sexes
    g1_mask = df[sex_col] == g1
    g2_mask = df[sex_col] == g2

    numeric = get_numeric_df(df)
    results = {}

    # ---------------- BINARIAS (χ² / Fisher) ----------------
    for col in numeric.columns:
        if col == sex_col:
            continue
        s = df[col]

        if not _is_binary(s):
            continue

        # tabla 2x2: valor=1 vs valor=0 en cada sexo
        a = int(((g1_mask) & (df[col] == 1)).sum())
        b = int(((g1_mask) & (df[col] == 0)).sum())
        c = int(((g2_mask) & (df[col] == 1)).sum())
        d = int(((g2_mask) & (df[col] == 0)).sum())

        if (a + b == 0) or (c + d == 0):
            continue

        table = [[a, b], [c, d]]

        try:
            chi2, p, dof, exp = stats.chi2_contingency(table)
            test_name = "chi2"
        except Exception:
            try:
                _, p = stats.fisher_exact(table)
                test_name = "fisher"
            except Exception:
                continue

        p1 = a / (a + b) if (a + b) > 0 else None
        p2 = c / (c + d) if (c + d) > 0 else None

        results[col] = {
            "type": "binary",
            "test_name": test_name,
            "p": float(p),
            "prop_sex_" + str(g1): p1,
            "prop_sex_" + str(g2): p2,
            "n": {str(g1): a + b, str(g2): c + d},
        }

    # ---------------- CONTINUAS (t-test / MW) ----------------
    for col in numeric.columns:
        if col == sex_col:
            continue
        if _is_binary(df[col]):
            # ya tratada arriba
            continue

        x = df[g1_mask][col].dropna().values
        y = df[g2_mask][col].dropna().values

        if len(x) < 3 or len(y) < 3:
            continue

        # Intento 1: Welch t-test
        try:
            t_stat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            test_name = "Welch t-test"
            mean_x = float(np.mean(x))
            mean_y = float(np.mean(y))
            diff = mean_x - mean_y

            # Varianzas y df de Welch
            sx2 = float(np.var(x, ddof=1))
            sy2 = float(np.var(y, ddof=1))
            nx = len(x)
            ny = len(y)
            se_diff = (sx2 / nx + sy2 / ny) ** 0.5

            ci_low = None
            ci_high = None
            if se_diff > 0:
                df_w = (sx2 / nx + sy2 / ny) ** 2 / (
                    (sx2 ** 2) / ((nx ** 2) * (nx - 1)) +
                    (sy2 ** 2) / ((ny ** 2) * (ny - 1))
                )
                t_crit = stats.t.ppf(0.975, df_w)
                ci_low = diff - t_crit * se_diff
                ci_high = diff + t_crit * se_diff
            else:
                df_w = None

            # Cohen's d (aprox. pooled SD)
            pooled_sd = None
            d_val = None
            if nx + ny > 2:
                pooled_sd = (((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)) ** 0.5
                if pooled_sd > 0:
                    d_val = diff / pooled_sd

            results[col] = {
                "type": "continuous",
                "test_name": test_name,
                "p": float(p),
                "effect": diff,          # mean_x - mean_y
                "ci_low": ci_low,
                "ci_high": ci_high,
                "cohens_d": d_val,
                "mean_sex_" + str(g1): mean_x,
                "mean_sex_" + str(g2): mean_y,
                "n": {str(g1): nx, str(g2): ny},
            }

        except Exception:
            # Fallback: Mann–Whitney
            try:
                u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                continue

            med_x = float(np.median(x))
            med_y = float(np.median(y))
            diff = med_x - med_y

            results[col] = {
                "type": "continuous",
                "test_name": "Mann-Whitney U",
                "p": float(p),
                "effect": diff,          # median_x - median_y
                "ci_low": None,
                "ci_high": None,
                "cohens_d": None,
                "median_sex_" + str(g1): med_x,
                "median_sex_" + str(g2): med_y,
                "n": {str(g1): len(x), str(g2): len(y)},
            }

    return results
