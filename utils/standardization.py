import pandas as pd
import numpy as np
from scipy import stats


# --- Statistical Standardization ---

def zscore_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        series = result[col].dropna()
        if series.std() == 0:
            result[col] = 0.0
        else:
            result[col] = (result[col] - series.mean()) / series.std()
    return result


def minmax_scale(df: pd.DataFrame, columns: list, feature_range=(0, 1)) -> pd.DataFrame:
    result = df.copy()
    lo, hi = feature_range
    for col in columns:
        series = result[col].dropna()
        col_min, col_max = series.min(), series.max()
        if col_max == col_min:
            result[col] = lo
        else:
            result[col] = lo + (result[col] - col_min) * (hi - lo) / (col_max - col_min)
    return result


# --- String Cleaning ---

def trim_whitespace(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        result[col] = result[col].astype(str).str.strip()
        result[col] = result[col].replace("nan", np.nan)
    return result


def change_case(df: pd.DataFrame, columns: list, case: str = "lower") -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if case == "lower":
            result[col] = result[col].astype(str).str.lower()
        elif case == "upper":
            result[col] = result[col].astype(str).str.upper()
        elif case == "title":
            result[col] = result[col].astype(str).str.title()
        result[col] = result[col].replace("nan", np.nan)
    return result


def normalize_dates(df: pd.DataFrame, columns: list, target_format: str = "%Y-%m-%d") -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        result[col] = pd.to_datetime(result[col], errors="coerce", dayfirst=True).dt.strftime(target_format)
    return result


# --- Common Cleaning ---

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def drop_missing_rows(df: pd.DataFrame, threshold_pct: float = 100.0) -> pd.DataFrame:
    thresh = int(len(df.columns) * (1 - threshold_pct / 100))
    return df.dropna(thresh=max(thresh, 1)).reset_index(drop=True)


def fill_missing(df: pd.DataFrame, columns: list, strategy: str = "mean") -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if strategy == "mean" and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].fillna(result[col].median())
        elif strategy == "mode":
            mode_val = result[col].mode()
            if not mode_val.empty:
                result[col] = result[col].fillna(mode_val.iloc[0])
        elif strategy == "ffill":
            result[col] = result[col].ffill()
        elif strategy == "bfill":
            result[col] = result[col].bfill()
        elif strategy == "zero":
            result[col] = result[col].fillna(0)
        elif strategy == "empty_string":
            result[col] = result[col].fillna("")
    return result
