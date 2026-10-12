import numpy as np
import pandas as pd

def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def convert_numeric_columns(df, exclude=None):
    df = df.copy()
    exclude = set(exclude or [])
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def missing_report(df):
    out = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean().mul(100)
    })
    return out.sort_values("missing_count", ascending=False)

def replace_missing_with_mean(df, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        mean = df[col].mean()
        if pd.notna(mean):
            df[col] = df[col].fillna(mean)
    return df

def replace_outliers_with_mean(df, numeric_cols, factor=1.5):
    """Assignment rule: replace IQR outliers with the corresponding column mean."""
    df = df.copy()
    report = []
    for col in numeric_cols:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            report.append({"column": col, "outliers": 0, "mean_used": s.mean()})
            continue
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (s < lower) | (s > upper)
        mean = s.mean()
        n = int(mask.sum())
        if pd.notna(mean):
            df.loc[mask, col] = mean
        report.append({"column": col, "outliers": n, "mean_used": mean,
                       "lower_bound": lower, "upper_bound": upper})
    return df, pd.DataFrame(report)

def clean_dataset(df):
    df = standardize_columns(df)

    date_cols = [c for c in ["Start", "End"] if c in df.columns]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    id_cols = [c for c in [
        "Bearer Id", "IMSI", "MSISDN/Number", "IMEI",
        "Start", "End", "Last Location Name",
        "Handset Manufacturer", "Handset Type"
    ] if c in df.columns]

    numeric_cols = [c for c in df.columns if c not in id_cols]
    df = convert_numeric_columns(df, exclude=id_cols)

    # IDs are required for grouping; rows with no customer ID cannot be used.
    if "MSISDN/Number" in df.columns:
        df = df[df["MSISDN/Number"].notna()].copy()

    df = df.drop_duplicates().reset_index(drop=True)

    # Fill categorical values with mode.
    categorical = [c for c in ["Last Location Name", "Handset Manufacturer", "Handset Type"]
                   if c in df.columns]
    for c in categorical:
        mode = df[c].mode(dropna=True)
        if len(mode):
            df[c] = df[c].fillna(mode.iloc[0])

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = replace_missing_with_mean(df, numeric_cols)
    df, outlier_report = replace_outliers_with_mean(df, numeric_cols)

    return df, outlier_report
