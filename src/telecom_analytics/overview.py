import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .config import HANDSET_MFR, HANDSET_TYPE, APP_PAIRS, FIGURES, USER_ID
from .features import customer_overview

def top_handsets(df, n=10):
    return df[HANDSET_TYPE].value_counts(dropna=False).head(n).reset_index(
        name="sessions"
    ).rename(columns={HANDSET_TYPE: "handset_type"})

def top_manufacturers(df, n=3):
    return df[HANDSET_MFR].value_counts(dropna=False).head(n).reset_index(
        name="sessions"
    ).rename(columns={HANDSET_MFR: "manufacturer"})

def top_handsets_by_manufacturer(df, top_n_mfr=3, top_n_handsets=5):
    manufacturers = df[HANDSET_MFR].value_counts().head(top_n_mfr).index
    rows = []
    for mfr in manufacturers:
        s = df.loc[df[HANDSET_MFR] == mfr, HANDSET_TYPE].value_counts().head(top_n_handsets)
        for handset, count in s.items():
            rows.append({HANDSET_MFR: mfr, HANDSET_TYPE: handset, "sessions": count})
    return pd.DataFrame(rows)

def descriptive_statistics(user_df):
    return user_df.select_dtypes(include=np.number).describe().T

def decile_analysis(user_df):
    x = user_df.copy()
    x["duration_decile"] = pd.qcut(
        x["session_duration_ms"].rank(method="first"),
        10, labels=False
    ) + 1
    out = x.groupby("duration_decile", as_index=False).agg(
        users=(USER_ID, "count"),
        total_data_bytes=("total_traffic_bytes", "sum"),
        average_duration_ms=("session_duration_ms", "mean")
    )
    # Assignment asks top five decile classes.
    return out.sort_values("duration_decile", ascending=False).head(5).sort_values("duration_decile")

def correlation(user_df):
    cols = [f"{a}_total" for a in APP_PAIRS]
    return user_df[cols].corr()

def run_pca(user_df, n_components=None):
    cols = [f"{a}_total" for a in APP_PAIRS]
    X = StandardScaler().fit_transform(user_df[cols])
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X)
    pca_df = pd.DataFrame(
        transformed,
        columns=[f"PC{i+1}" for i in range(transformed.shape[1])]
    )
    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
    })
    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    return pca_df, explained, loadings

def save_overview_charts(df, user_df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    top_handsets(df).plot.barh(x="handset_type", y="sessions", legend=False)
    plt.tight_layout(); plt.savefig(FIGURES/"top_10_handsets.png", dpi=160); plt.close()

    apps = [f"{a}_total" for a in APP_PAIRS]
    user_df[apps].sum().sort_values(ascending=False).plot.bar()
    plt.ylabel("Bytes"); plt.title("Total application traffic")
    plt.tight_layout(); plt.savefig(FIGURES/"application_traffic.png", dpi=160); plt.close()

    corr = correlation(user_df)
    plt.figure(figsize=(9,7))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.tight_layout(); plt.savefig(FIGURES/"application_correlation.png", dpi=160); plt.close()
