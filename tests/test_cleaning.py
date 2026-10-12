import pandas as pd
from src.telecom_analytics.cleaning import replace_missing_with_mean, replace_outliers_with_mean

def test_missing_mean():
    df = pd.DataFrame({"x":[1.0,2.0,None,4.0]})
    out = replace_missing_with_mean(df, ["x"])
    assert out["x"].isna().sum() == 0
    assert abs(out.loc[2,"x"] - df["x"].mean()) < 1e-9

def test_outlier_mean():
    df = pd.DataFrame({"x":[1.,2.,3.,4.,1000.]})
    out, report = replace_outliers_with_mean(df, ["x"])
    assert out["x"].max() < 1000
    assert int(report.loc[0,"outliers"]) >= 1
