from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENCY_PATTERN = re.compile(r"[,$€£₹]")

@dataclass
class CleanResult:
    df: pd.DataFrame
    duplicates_removed: int
    rows_before: int
    rows_after: int
    notes: List[str]


def _note(notes: List[str], msg: str):
    notes.append(msg)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _rename_and_select(df: pd.DataFrame, cfg: Dict, notes: List[str]) -> pd.DataFrame:
    std = cfg.get("standardize", {})
    if std.get("normalize_column_names", True):
        df = _normalize_cols(df)
        _note(notes, "Standardized column names (lowercase, underscores).")
    rename_map = std.get("rename_map", {})
    if rename_map:
        # map keys should already be the original names; normalize for safety
        norm_map = {k.strip().lower().replace(" ", "_"): v for k, v in rename_map.items()}
        df = df.rename(columns=norm_map)
        _note(notes, f"Renamed columns via map: {list(norm_map.values())}.")
    select_cols = std.get("select_columns")
    if select_cols:
        missing = [c for c in select_cols if c not in df.columns]
        if missing:
            _note(notes, f"Select columns requested but missing: {missing}")
        keep = [c for c in select_cols if c in df.columns]
        df = df[keep]
        _note(notes, f"Selected columns: {keep}.")
    return df


def _parse_dates_numbers(df: pd.DataFrame, cfg: Dict, notes: List[str]) -> pd.DataFrame:
    parsing = cfg.get("parsing", {})
    date_cfg = parsing.get("date_columns", {})
    for col, rules in date_cfg.items():
        if col in df.columns:
            formats = rules.get("formats", None)
            coerce = rules.get("coerce_invalid_to_null", True)
            if formats:
                # try multiple formats
                ser = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                for fmt in formats:
                    parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
                    ser = ser.fillna(parsed)
                df[col] = ser if coerce else pd.to_datetime(df[col], errors="ignore")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            _note(notes, f"Parsed dates in '{col}'.")
    # numeric hints
    hints = parsing.get("numeric_hints", ["amount","price","sales","revenue","total","cost"])
    for col in df.columns:
        if any(h in col for h in hints):
            df[col] = (
                df[col].astype(str)
                .str.replace(CURRENCY_PATTERN, "", regex=True)
                .str.replace(" ", "", regex=False)
            )
            # handle decimal separators
            thousand_seps = parsing.get("thousand_separators", [","," "])
            dec_seps = parsing.get("decimal_separators", [".",","])
            # naive normalization: remove thousands separators, prefer dot as decimal
            for sep in thousand_seps:
                df[col] = df[col].str.replace(sep, "", regex=False)
            # if comma appears but dot not, replace comma with dot
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            _note(notes, f"Parsed numeric-like values in '{col}'.")
    return df


def _deduplicate(df: pd.DataFrame, cfg: Dict, notes: List[str]) -> Tuple[pd.DataFrame, int]:
    dd = cfg.get("deduplication", {"enabled": True, "keep": "first"})
    if not dd.get("enabled", True):
        return df, 0
    subset = dd.get("subset_keys", None)
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=dd.get("keep", "first"))
    removed = before - len(df)
    _note(notes, f"Removed {removed} duplicate rows (subset={subset}).")
    return df, removed


def _missing_values(df: pd.DataFrame, cfg: Dict, notes: List[str]) -> pd.DataFrame:
    mv = cfg.get("missing_values", {})
    # drop conditions first
    for col in mv.get("drop_rows_if_any_null_in", []):
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            _note(notes, f"Dropped {before-len(df)} rows where '{col}' was null.")
    if mv.get("drop_rows_if_all_null_in", []):
        cols = [c for c in mv["drop_rows_if_all_null_in"] if c in df.columns]
        if cols:
            before = len(df)
            df = df.dropna(subset=cols, how="all")
            _note(notes, f"Dropped {before-len(df)} rows where all {cols} were null.")
    # fill
    strategy_num = mv.get("strategy", {}).get("numeric", "median")
    strategy_cat = mv.get("strategy", {}).get("categorical", "mode")
    per_fill = mv.get("per_column_fill", {})
    # numeric
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        if strategy_num == "median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif strategy_num == "mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif strategy_num == "zero":
            df[num_cols] = df[num_cols].fillna(0)
        _note(notes, f"Filled numeric NaNs using '{strategy_num}' for: {num_cols}.")
    # categorical
    obj_cols = df.select_dtypes(exclude="number").columns.tolist()
    for c in obj_cols:
        if c in per_fill:
            df[c] = df[c].fillna(per_fill[c])
        else:
            mode = df[c].mode(dropna=True)
            if strategy_cat == "mode" and not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
            elif strategy_cat == "constant":
                df[c] = df[c].fillna("Unknown")
    _note(notes, f"Filled categorical NaNs using '{strategy_cat}'/constants where specified.")
    return df


def _cast_types(df: pd.DataFrame, cfg: Dict, notes: List[str]) -> pd.DataFrame:
    cast = cfg.get("casting", {}).get("dtypes", {})
    safe = cfg.get("casting", {}).get("safe_cast", True)
    for col, typ in cast.items():
        if col not in df.columns:
            continue
        try:
            if typ == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce" if safe else "raise")
            else:
                df[col] = df[col].astype(typ)
            _note(notes, f"Casted '{col}' to {typ}.")
        except Exception as e:
            if safe:
                _note(notes, f"Safe cast failed for '{col}'→{typ}: {e}")
            else:
                raise
    return df


def _constraints(df: pd.DataFrame, cfg: Dict, notes: List[str]) -> pd.DataFrame:
    cons = cfg.get("constraints", {})
    # non-negative
    neg_cfg = cfg.get("negatives", {"handle": "set_null", "columns": []})
    handle_neg = neg_cfg.get("handle", "set_null")
    neg_cols = neg_cfg.get("columns", cons.get("non_negative", []))
    for col in neg_cols:
        if col in df.columns:
            mask = df[col] < 0
            if mask.any():
                if handle_neg == "drop_row":
                    before = len(df)
                    df = df.loc[~mask].copy()
                    _note(notes, f"Dropped {before-len(df)} rows with negative '{col}'.")
                elif handle_neg == "abs":
                    df.loc[mask, col] = df.loc[mask, col].abs()
                    _note(notes, f"Converted negatives to abs in '{col}'.")
                elif handle_neg == "ignore":
                    pass
                else:  # set_null
                    df.loc[mask, col] = np.nan
                    _note(notes, f"Set negatives to null in '{col}'.")
    # computed checks: sales_amount = qty * price
    comp = cons.get("computed_checks", {}).get("sales_amount_from_fields", {})
    if comp.get("enabled", False):
        q = comp.get("quantity_col")
        p = comp.get("unit_price_col")
        t = comp.get("target_col")
        if all(c in df.columns for c in [q,p,t]):
            calc = df[q] * df[p]
            # overwrite if deviation > threshold or target is null
            thresh = comp.get("overwrite_if_deviation_pct_gt", 5)
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = np.abs(calc - df[t]) / np.where(df[t]==0, 1, df[t]) * 100
            overwrite = df[t].isna() | (dev > thresh)
            n = int(overwrite.sum())
            df.loc[overwrite, t] = calc.loc[overwrite]
            _note(notes, f"Computed '{t}' from {q}*{p} for {n} rows (>{thresh}% deviation or null).")
    return df


def _outliers(df: pd.DataFrame, cfg: Dict, notes: List[str]) -> pd.DataFrame:
    out = cfg.get("outliers", {"method": "iqr", "handle": "cap"})
    if out.get("method", "iqr") != "iqr":
        return df
    cols = [c for c in out.get("columns", []) if c in df.columns]
    if not cols:
        return df
    mult = out.get("iqr_multiplier", 1.5)
    handle = out.get("handle", "cap")
    if handle == "cap" and out.get("cap_limits", "quantiles") == "quantiles":
        lower_q = out.get("cap_lower_q", 0.01)
        upper_q = out.get("cap_upper_q", 0.99)
    for c in cols:
        series = df[c]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - mult * iqr
        high = q3 + mult * iqr
        mask = (series < low) | (series > high)
        if not mask.any():
            continue
        if handle == "drop":
            before = len(df)
            df = df.loc[~mask].copy()
            _note(notes, f"Dropped {before-len(df)} outliers in '{c}'.")
        elif handle == "flag":
            flag_col = c + cfg.get("outliers", {}).get("flag_column_suffix", "_is_outlier")
            df[flag_col] = mask.astype(int)
            _note(notes, f"Flagged outliers in '{c}' as '{flag_col}'.")
        elif handle == "cap":
            if out.get("cap_limits", "quantiles") == "quantiles":
                lo_cap = series.quantile(lower_q)
                hi_cap = series.quantile(upper_q)
            else:
                lo_cap, hi_cap = low, high
            df.loc[series < low, c] = lo_cap
            df.loc[series > high, c] = hi_cap
            _note(notes, f"Capped outliers in '{c}' to [{lo_cap:.3g}, {hi_cap:.3g}].")
        else:
            # ignore
            pass
    return df


def _save_table_png(df: pd.DataFrame, path: Path, title: str, rows: int = 12) -> None:
    fig = plt.figure()
    plt.axis("off")
    sample = df.head(rows)
    plt.table(cellText=sample.values, colLabels=sample.columns, loc="center")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def clean_with_config(df: pd.DataFrame, cfg: Dict) -> CleanResult:
    notes: List[str] = []
    rows_before = len(df)

    # 1) standardize/rename/select
    df = _rename_and_select(df, cfg, notes)

    # 2) parsing
    df = _parse_dates_numbers(df, cfg, notes)

    # 3) dedup
    df, removed = _deduplicate(df, cfg, notes)

    # 4) missing
    df = _missing_values(df, cfg, notes)

    # 5) casting
    df = _cast_types(df, cfg, notes)

    # 6) constraints (negatives handling, computed checks)
    df = _constraints(df, cfg, notes)

    # 7) outliers
    df = _outliers(df, cfg, notes)

    rows_after = len(df)
    return CleanResult(df=df, duplicates_removed=removed, rows_before=rows_before, rows_after=rows_after, notes=notes)


def run_cleaning(input_csv: Path, outdir: Path, config: Optional[Dict] = None) -> None:
    """
    YAML-driven cleaning runner. If config is None, uses a minimal safe default path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv)
    _save_table_png(df_raw, outdir / (config.get("io", {}).get("before_png", "sample_before.png") if config else "sample_before.png"), "BEFORE — Raw Sample")

    if config is None:
        config = {}

    result = clean_with_config(df_raw, config)

    cleaned_name = config.get("io", {}).get("cleaned_filename", "cleaned.csv")
    result.df.to_csv(outdir / cleaned_name, index=False)

    _save_table_png(result.df, outdir / (config.get("io", {}).get("after_png", "sample_after.png") if config else "sample_after.png"), "AFTER — Cleaned Sample")

    log_path = outdir / (config.get("io", {}).get("log_file", "cleaning_log.txt") if config else "cleaning_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Rows before: {result.rows_before}\n")
        f.write(f"Rows after: {result.rows_after}\n")
        f.write(f"Duplicates removed: {result.duplicates_removed}\n")
        for line in result.notes:
            f.write(f"- {line}\n")

    print(f"Saved outputs to: {outdir}")
