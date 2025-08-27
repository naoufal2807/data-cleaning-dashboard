from __future__ import annotations
import numpy as np
import pandas as pd

def missing_by_column(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({"missing": miss, "missing_%": pct})
    out.index.name = "column"
    return out.reset_index()

def overall_missing_pct(df: pd.DataFrame) -> float:
    return float((df.isna().sum().sum() / (df.shape[0]*df.shape[1])) * 100)

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc["missing"] = num.isna().sum()
    return desc.reset_index().rename(columns={"index": "column"})

def outlier_counts_iqr(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    rows = []
    for c in num.columns:
        q1 = num[c].quantile(0.25)
        q3 = num[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5*iqr
        high = q3 + 1.5*iqr
        mask = (num[c] < low) | (num[c] > high)
        rows.append({"column": c, "outliers": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("outliers", ascending=False)

def negative_counts(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame(columns=["column","negatives"])
    neg = (num < 0).sum()
    return pd.DataFrame({"column": neg.index, "negatives": neg.values}).sort_values("negatives", ascending=False)
