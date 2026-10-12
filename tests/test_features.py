import pandas as pd
from src.telecom_analytics.features import customer_overview

def test_customer_aggregation():
    df = pd.DataFrame({
        "MSISDN/Number":[1,1,2],
        "Dur. (ms)":[100,200,300],
        "Total DL (Bytes)":[10,20,30],
        "Total UL (Bytes)":[1,2,3],
        "Social Media DL (Bytes)":[1,2,3],
        "Social Media UL (Bytes)":[1,1,1],
        "Google DL (Bytes)":[1,1,1],
        "Google UL (Bytes)":[1,1,1],
        "Email DL (Bytes)":[1,1,1],
        "Email UL (Bytes)":[1,1,1],
        "Youtube DL (Bytes)":[1,1,1],
        "Youtube UL (Bytes)":[1,1,1],
        "Netflix DL (Bytes)":[1,1,1],
        "Netflix UL (Bytes)":[1,1,1],
        "Gaming DL (Bytes)":[1,1,1],
        "Gaming UL (Bytes)":[1,1,1],
        "Other DL (Bytes)":[1,1,1],
        "Other UL (Bytes)":[1,1,1],
    })
    out = customer_overview(df)
    row = out[out["MSISDN/Number"] == 1].iloc[0]
    assert row["xdr_sessions"] == 2
    assert row["session_duration_ms"] == 300
    assert row["total_traffic_bytes"] == 33
