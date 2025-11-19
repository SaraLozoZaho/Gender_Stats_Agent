# analysis/descriptive_gender.py

import pandas as pd
import numpy as np

NUMERIC_DTYPES = ["float64", "int64", "float32", "int32"]


def get_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=NUMERIC_DTYPES)


def compute_descriptive_stats_gender(df: pd.DataFrame, sex_col: str) -> dict:
    """
    Compute basic descriptive statistics per sex group (1=male, 2=female),
    similar to the respiratory agent's structure.
    """

    numeric = get_numeric_df(df)
    overall = numeric.describe().T.round(4).to_dict(orient="index")

    by_sex = {}

    if sex_col in df.columns:
        for sex, sub in df.groupby(sex_col):
            if pd.isna(sex):
                continue
            desc = sub[numeric.columns].describe().T.round(4).to_dict(orient="index")
            by_sex[str(sex)] = desc

    return {
        "overall": overall,
        "by_sex": by_sex,
        "numeric_columns": list(numeric.columns)
    }
