import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

def scale_features(df, cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    return X, scaler

def elbow(X, k_range=range(2, 11), sample_size=5000):
    """Efficient elbow analysis. Silhouette is calculated on a sample because
    the exact calculation is O(n^2) and is impractical for 100k+ customers."""
    rows = []
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[sample_idx]
    for k in k_range:
        model = MiniBatchKMeans(
            n_clusters=k, random_state=42, n_init=5,
            batch_size=2048, max_iter=100
        )
        model.fit(X)
        sample_labels = model.predict(X_sample)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_sample": float(silhouette_score(X_sample, sample_labels))
        })
    return pd.DataFrame(rows)

def kmeans_clusters(df, cols, k):
    X, scaler = scale_features(df, cols)
    model = MiniBatchKMeans(
        n_clusters=k, random_state=42, n_init=10,
        batch_size=2048, max_iter=200
    )
    labels = model.fit_predict(X)
    out = df.copy()
    out["cluster"] = labels
    return out, model, scaler, X

def kmeans_clusters(df, cols, k):
    X, scaler = scale_features(df, cols)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    out = df.copy()
    out["cluster"] = labels
    return out, model, scaler, X

def cluster_summary(clustered, cols):
    return clustered.groupby("cluster")[cols].agg(["min","max","mean","sum"]).round(3)
