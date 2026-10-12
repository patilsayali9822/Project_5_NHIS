import numpy as np
import pandas as pd
from .config import USER_ID, DURATION, APP_PAIRS, EXPERIENCE_COLS, HANDSET_TYPE

def add_application_totals(df):
    df = df.copy()
    for app, (dl, ul) in APP_PAIRS.items():
        df[f"{app}_total"] = df[dl].fillna(0) + df[ul].fillna(0)
    return df

def customer_overview(df):
    df = add_application_totals(df)
    agg = {
        "xdr_sessions": (USER_ID, "size"),
        "session_duration_ms": (DURATION, "sum"),
        "total_dl_bytes": ("Total DL (Bytes)", "sum"),
        "total_ul_bytes": ("Total UL (Bytes)", "sum"),
    }
    for app in APP_PAIRS:
        agg[f"{app}_total"] = (f"{app}_total", "sum")
    out = df.groupby(USER_ID).agg(**agg).reset_index()
    out["total_traffic_bytes"] = out["total_dl_bytes"] + out["total_ul_bytes"]
    return out

def engagement_features(df):
    return customer_overview(df)[[
        USER_ID, "xdr_sessions", "session_duration_ms", "total_traffic_bytes"
    ]]

def experience_features(df):
    work = df.copy()
    work["tcp_retransmission_bytes"] = (
        work[EXPERIENCE_COLS["tcp_dl"]].fillna(0) +
        work[EXPERIENCE_COLS["tcp_ul"]].fillna(0)
    )
    work["rtt_ms"] = (
        work[EXPERIENCE_COLS["rtt_dl"]].fillna(0) +
        work[EXPERIENCE_COLS["rtt_ul"]].fillna(0)
    ) / 2
    work["throughput_kbps"] = (
        work[EXPERIENCE_COLS["tp_dl"]].fillna(0) +
        work[EXPERIENCE_COLS["tp_ul"]].fillna(0)
    ) / 2

    # Mode is the assignment's requested treatment for categorical handset type.
    mode = work[HANDSET_TYPE].mode(dropna=True)
    if len(mode):
        work[HANDSET_TYPE] = work[HANDSET_TYPE].fillna(mode.iloc[0])

    out = work.groupby(USER_ID).agg(
        avg_tcp_retransmission=("tcp_retransmission_bytes", "mean"),
        avg_rtt=("rtt_ms", "mean"),
        avg_throughput=("throughput_kbps", "mean"),
        handset_type=(HANDSET_TYPE, lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    ).reset_index()
    return out

def build_user_feature_store(df):
    ov = customer_overview(df)
    ex = experience_features(df)
    return ov.merge(ex, on=USER_ID, how="inner")
