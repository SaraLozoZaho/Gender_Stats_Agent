# GENDER_STATS_AGENT/agent/gender_agent_core.py

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from utils.sex_gender_utils import detect_sex_gender_vars


# ======================================================================
# Helpers
# ======================================================================

def _numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _has_variability(arr: np.ndarray):
    arr = arr[~np.isnan(arr)]
    return len(np.unique(arr)) >= 2


# ======================================================================
# 1) DESCRIPTIVES
# ======================================================================

def compute_descriptives(df: pd.DataFrame, sex_gender_info):
    result = {
        "by_sex": {},
        "by_gender": {},
        "overall": {}
    }

    numeric_cols = _numeric_columns(df)
    overall = {}
    for var in numeric_cols:
        x = df[var].dropna()
        if len(x) == 0:
            continue
        overall[var] = {
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "n": int(len(x)),
        }
    result["overall"] = overall

    # --- SEX ---
    if sex_gender_info.sex_col:
        col = sex_gender_info.sex_col
        for g, sub in df.groupby(col):
            desc = {}
            for var in numeric_cols:
                x = sub[var].dropna()
                if len(x) == 0:
                    continue
                desc[var] = {
                    "mean": float(np.mean(x)),
                    "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
                    "n": int(len(x)),
                }
            result["by_sex"][str(g)] = desc

    # --- GENDER (por si algún día lo tienes) ---
    if sex_gender_info.gender_col:
        col = sex_gender_info.gender_col
        for g, sub in df.groupby(col):
            desc = {}
            for var in numeric_cols:
                x = sub[var].dropna()
                if len(x) == 0:
                    continue
                desc[var] = {
                    "mean": float(np.mean(x)),
                    "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
                    "n": int(len(x)),
                }
            result["by_gender"][str(g)] = desc

    return result


# ======================================================================
# 2) TESTS POR GRUPOS (SEX) – t-test / Mann–Whitney + efecto + CI
# ======================================================================

def compute_group_tests(df: pd.DataFrame, sex_gender_info):
    result = {}

    group_col = sex_gender_info.sex_col or sex_gender_info.gender_col
    if not group_col:
        return result

    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return result

    g1, g2 = groups
    numeric_cols = _numeric_columns(df)

    for var in numeric_cols:
        x = df[df[group_col] == g1][var].dropna().values
        y = df[df[group_col] == g2][var].dropna().values

        if len(x) < 3 or len(y) < 3:
            continue

        # Normalidad aproximada
        normal = False
        try:
            _, p1 = stats.shapiro(x)
            _, p2 = stats.shapiro(y)
            normal = (p1 > 0.05 and p2 > 0.05)
        except Exception:
            pass

        if normal:
            test_name = "t-test (Welch)"
            t, p = stats.ttest_ind(x, y, equal_var=False)
            effect = float(np.mean(x) - np.mean(y))  # diferencia de medias
            # Cohen's d aproximado
            sp = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) /
                         (len(x) + len(y) - 2))
            d = effect / sp if sp > 0 else None
        else:
            test_name = "Mann-Whitney U"
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            effect = float(np.median(x) - np.median(y))
            d = None

        # Bootstrap CI de la diferencia de medias
        ci = None
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
            pass

        result[var] = {
            "test_name": test_name,
            "p": float(p),
            "effect": effect,
            "ci": ci,
            "cohens_d": float(d) if d is not None else None,
            "group_labels": [str(g1), str(g2)],
            "n1": int(len(x)),
            "n2": int(len(y)),
        }

    return result


# ======================================================================
# 3) ÍNDICES DERIVADOS (VT indexado por peso)
# ======================================================================

def compute_derived_indices(df: pd.DataFrame):
    """
    Crea índices derivados, por ahora:
    - VT_per_ideal_weight si existen columnas tipo VT y IdealWeight
    - VT_per_weight (por kg real) si existen VT y Weight
    Devuelve un dict resumen y NO modifica el df original.
    """

    derived = {}

    # Intentar encontrar columnas razonables
    cols_lower = {c.lower(): c for c in df.columns}

    vt_col = None
    for key in cols_lower:
        if "vt" in key or "tidal" in key:
            vt_col = cols_lower[key]
            break

    ideal_col = None
    for key in cols_lower:
        if "ideal" in key and "weight" in key:
            ideal_col = cols_lower[key]
            break

    weight_col = None
    for key in cols_lower:
        if key in ["weight", "bodyweight", "gewicht"]:
            weight_col = cols_lower[key]
            break

    # Crear índices en una copia
    df_idx = df.copy()

    if vt_col and ideal_col:
        df_idx["VT_per_ideal_weight"] = df_idx[vt_col] / df_idx[ideal_col]
        x = df_idx["VT_per_ideal_weight"].dropna()
        if len(x) > 0:
            derived["VT_per_ideal_weight"] = {
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
                "n": int(len(x)),
            }

    if vt_col and weight_col:
        df_idx["VT_per_weight"] = df_idx[vt_col] / df_idx[weight_col]
        x = df_idx["VT_per_weight"].dropna()
        if len(x) > 0:
            derived["VT_per_weight"] = {
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
                "n": int(len(x)),
            }

    return derived


# ======================================================================
# 4) MODELOS MULTIVARIABLES (lineal + logístico si hay outcome)
# ======================================================================

def compute_multivariable_models(df: pd.DataFrame, sex_gender_info):
    """
    Modelos ajustados:
    - Linear: outcome continuo tipo 'Compliance' ~ Sex + Age + Height + WeightAboveIdeal + Smoking
    - Logistic: primer outcome binario cuyo nombre contenga 'outcome'
    """
    models = {}

    sex_col = sex_gender_info.sex_col or sex_gender_info.gender_col
    if not sex_col:
        return models

    # ------------------------------------------------------------------
    # 4.1. LINEAR – buscar columna de compliance
    # ------------------------------------------------------------------
    candidates = [c for c in df.columns if "compliance" in c.lower()]
    if candidates:
        outcome_col = candidates[0]
        cols = [sex_col]

        for label in ["age", "height", "weight above ideal", "weight_above_ideal",
                      "weightaboveideal", "weight", "smoking", "raucher", "smoker"]:
            for c in df.columns:
                if label in c.lower() and c not in cols:
                    cols.append(c)

        # Quitar duplicados
        cols = list(dict.fromkeys(cols))

        # Construir dataset limpio
        used_cols = [outcome_col] + cols
        sub = df[used_cols].dropna()
        if len(sub) >= 10:
            y = sub[outcome_col].astype(float)
            X = sub[cols].copy()

            # Codificar categóricas simples (e.g. fumador 0/1 se queda igual)
            for c in X.columns:
                if not np.issubdtype(X[c].dtype, np.number):
                    X[c] = X[c].astype("category").cat.codes

            X = sm.add_constant(X)

            try:
                model = sm.OLS(y, X).fit()
                n = int(model.nobs)
                r2_adj = float(model.rsquared_adj)

                sex_term = None
                for name in model.params.index:
                    if name == sex_col or name.endswith(f"[T.1]") or name.endswith(f"[T.2]"):
                        sex_term = name
                        break

                sex_effect = None
                if sex_term:
                    beta = float(model.params[sex_term])
                    p = float(model.pvalues[sex_term])
                    sex_effect = f"beta={beta:.3f}, p={p:.3e}"

                models["linear_compliance"] = {
                    "metric": f"Adj R²={r2_adj:.3f}, n={n}",
                    "sex_effect": sex_effect,
                    "notes": f"Outcome={outcome_col}; covariates={cols}",
                }
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 4.2. LOGISTIC – outcome binario con 'outcome' en el nombre
    # ------------------------------------------------------------------
    bin_candidates = []
    for c in df.columns:
        if "outcome" in c.lower():
            # comprobar si es binaria 0/1
            vals = df[c].dropna().unique()
            if len(vals) == 2:
                bin_candidates.append(c)

    if bin_candidates:
        outcome_col = bin_candidates[0]
        cols = [sex_col]

        for label in ["age", "height", "weight", "smoking", "raucher", "smoker"]:
            for c in df.columns:
                if label in c.lower() and c not in cols:
                    cols.append(c)
        cols = list(dict.fromkeys(cols))

        sub = df[[outcome_col] + cols].dropna()
        if len(sub) >= 20:
            y = sub[outcome_col]
            if not np.issubdtype(y.dtype, np.number):
                y = y.astype("category").cat.codes

            X = sub[cols].copy()
            for c in X.columns:
                if not np.issubdtype(X[c].dtype, np.number):
                    X[c] = X[c].astype("category").cat.codes

            X = sm.add_constant(X)

            try:
                model = sm.Logit(y, X).fit(disp=False)
                n = int(model.nobs)

                sex_term = None
                for name in model.params.index:
                    if name == sex_col or name.endswith(f"[T.1]") or name.endswith(f"[T.2]"):
                        sex_term = name
                        break

                sex_effect = None
                if sex_term:
                    OR = float(np.exp(model.params[sex_term]))
                    p = float(model.pvalues[sex_term])
                    sex_effect = f"OR={OR:.3f}, p={p:.3e}"

                models["logistic_outcome"] = {
                    "metric": f"LogLik={model.llf:.1f}, n={n}",
                    "sex_effect": sex_effect,
                    "notes": f"Outcome={outcome_col}; covariates={cols}",
                }
            except Exception:
                pass

    return models


# ======================================================================
# 5) MAIN PIPELINE: build_gender_stats_payload
# ======================================================================

def build_gender_stats_payload(df: pd.DataFrame) -> dict:
    # Detectar columnas de sexo/género
    sex_gender_info = detect_sex_gender_vars(df)

    stats_json = {
        "dataset_info": {
            "n_total": int(len(df)),
            "count_by_sex": {},
            "count_by_gender": {},
        },
        "descriptive": {},
        "group_tests": {},
        "models": {},
        "gender_gap_indicators": {},
        "derived_indices": {},
        "sex_gender_info": {
            "sex_col": sex_gender_info.sex_col,
            "gender_col": sex_gender_info.gender_col,
        },
    }

    # Conteos
    if sex_gender_info.sex_col:
        stats_json["dataset_info"]["count_by_sex"] = (
            df[sex_gender_info.sex_col].value_counts(dropna=True).to_dict()
        )

    if sex_gender_info.gender_col:
        stats_json["dataset_info"]["count_by_gender"] = (
            df[sex_gender_info.gender_col].value_counts(dropna=True).to_dict()
        )

    # Descriptivos
    stats_json["descriptive"] = compute_descriptives(df, sex_gender_info)

    # Tests por grupo
    stats_json["group_tests"] = compute_group_tests(df, sex_gender_info)

    # Modelos multivariables
    stats_json["models"] = compute_multivariable_models(df, sex_gender_info)

    # Índices derivados (VT indexado, etc.)
    stats_json["derived_indices"] = compute_derived_indices(df)

    # Indicador simple de cuántas variables tienen p < 0.05
    try:
        sig = [
            v for v in stats_json["group_tests"].values()
            if isinstance(v, dict) and v.get("p") is not None and v["p"] < 0.05
        ]
        stats_json["gender_gap_indicators"]["significant_differences"] = len(sig)
    except Exception:
        pass

    return stats_json
