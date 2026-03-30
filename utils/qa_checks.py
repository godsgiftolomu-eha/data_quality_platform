import pandas as pd
import numpy as np
from scipy import stats


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isnull().sum()
    percent = (total / len(df) * 100).round(2)
    report = pd.DataFrame({
        "Column": total.index,
        "Missing Count": total.values,
        "Missing %": percent.values,
        "Total Rows": len(df),
    })
    return report.sort_values("Missing %", ascending=False).reset_index(drop=True)


def duplicate_report(df: pd.DataFrame) -> dict:
    dup_count = df.duplicated().sum()
    dup_rows = df[df.duplicated(keep=False)]
    return {
        "total_rows": len(df),
        "duplicate_rows": int(dup_count),
        "duplicate_percent": round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0,
        "duplicate_df": dup_rows,
    }


def datatype_report(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        inferred = pd.api.types.infer_dtype(df[col], skipna=True)
        records.append({
            "Column": col,
            "Pandas Dtype": str(df[col].dtype),
            "Inferred Type": inferred,
            "Unique Values": df[col].nunique(),
            "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A",
        })
    return pd.DataFrame(records)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns
    records = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_outliers = ((series < lower) | (series > upper)).sum()

        z_scores = np.abs(stats.zscore(series))
        z_outliers = (z_scores > 3).sum()

        records.append({
            "Column": col,
            "Total Values": len(series),
            "IQR Outliers": int(iqr_outliers),
            "IQR Outlier %": round(iqr_outliers / len(series) * 100, 2),
            "Z-Score Outliers (>3)": int(z_outliers),
            "Z-Score Outlier %": round(z_outliers / len(series) * 100, 2),
            "IQR Lower Bound": round(lower, 4),
            "IQR Upper Bound": round(upper, 4),
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


def distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns
    records = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        records.append({
            "Column": col,
            "Count": len(series),
            "Mean": round(series.mean(), 4),
            "Median": round(series.median(), 4),
            "Std Dev": round(series.std(), 4),
            "Min": round(series.min(), 4),
            "Max": round(series.max(), 4),
            "Skewness": round(series.skew(), 4),
            "Kurtosis": round(series.kurtosis(), 4),
            "25th %ile": round(series.quantile(0.25), 4),
            "75th %ile": round(series.quantile(0.75), 4),
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


def uniqueness_report(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        n_unique = df[col].nunique()
        records.append({
            "Column": col,
            "Unique Values": n_unique,
            "Total (non-null)": df[col].notna().sum(),
            "Cardinality %": round(n_unique / df[col].notna().sum() * 100, 2) if df[col].notna().sum() > 0 else 0,
            "Is Unique": n_unique == df[col].notna().sum(),
        })
    return pd.DataFrame(records).sort_values("Cardinality %", ascending=False).reset_index(drop=True)


def completeness_score(df: pd.DataFrame) -> float:
    total_cells = df.shape[0] * df.shape[1]
    filled = total_cells - df.isnull().sum().sum()
    return round(filled / total_cells * 100, 2) if total_cells > 0 else 0
