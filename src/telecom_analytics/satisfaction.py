import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .clustering import kmeans_clusters
from .config import USER_ID

def _distance_to_centroid(X, labels, centroid):
    return np.linalg.norm(X - centroid, axis=1)

def build_scores(user_df):
    engagement_cols = ["xdr_sessions", "session_duration_ms", "total_traffic_bytes"]
    experience_cols = ["avg_tcp_retransmission", "avg_rtt", "avg_throughput"]

    # Engagement: higher is better. We identify the least-engaged cluster by
    # the smallest mean of normalized engagement features.
    e_clustered, e_model, e_scaler, e_X = kmeans_clusters(user_df, engagement_cols, 3)
    e_means = e_clustered.groupby("cluster")[engagement_cols].mean()
    e_means_scaled = pd.DataFrame(
        e_scaler.transform(e_means),
        index=e_means.index,
        columns=engagement_cols
    ).mean(axis=1)
    least_cluster = e_means_scaled.idxmin()
    e_score = _distance_to_centroid(
        e_X, e_clustered["cluster"].to_numpy(), e_model.cluster_centers_[least_cluster]
    )

    # Experience: lower retransmission/RTT and higher throughput are better.
    # Transform throughput to a "badness" feature so all three dimensions have
    # the same direction: larger = worse experience.
    exp_work = user_df[experience_cols].copy()
    exp_work["throughput_badness"] = -exp_work.pop("avg_throughput")
    exp_work = exp_work.rename(columns={
        "avg_tcp_retransmission": "tcp_badness",
        "avg_rtt": "rtt_badness"
    })
    xcols = ["tcp_badness", "rtt_badness", "throughput_badness"]
    x_clustered, x_model, x_scaler, x_X = kmeans_clusters(
        pd.concat([user_df[[USER_ID]], exp_work], axis=1), xcols, 3
    )
    x_means = x_clustered.groupby("cluster")[xcols].mean()
    # Compare clusters in standardized feature space so TCP, RTT and
    # throughput have comparable influence.
    x_means_scaled = pd.DataFrame(
        x_scaler.transform(x_means),
        index=x_means.index,
        columns=xcols
    )
    worst_cluster = x_means_scaled.mean(axis=1).idxmax()
    x_score = _distance_to_centroid(
        x_X, x_clustered["cluster"].to_numpy(), x_model.cluster_centers_[worst_cluster]
    )

    scores = user_df[[USER_ID]].copy()
    scores["engagement_score"] = e_score
    scores["experience_score"] = x_score
    scores["satisfaction_score"] = (
        scores["engagement_score"] + scores["experience_score"]
    ) / 2

    # Normalize scores to 0-100 for dashboard readability while retaining raw scores.
    for c in ["engagement_score", "experience_score", "satisfaction_score"]:
        mn, mx = scores[c].min(), scores[c].max()
        scores[c + "_100"] = 100 * (scores[c] - mn) / (mx - mn) if mx > mn else 0

    scores["engagement_cluster"] = e_clustered["cluster"].values
    scores["experience_cluster"] = x_clustered["cluster"].values

    return scores, {
        "engagement_model": e_model,
        "engagement_scaler": e_scaler,
        "experience_model": x_model,
        "experience_scaler": x_scaler,
        "least_engaged_cluster": int(least_cluster),
        "worst_experience_cluster": int(worst_cluster)
    }

def satisfaction_regression(scores):
    # Do not use the target or its component scores as predictors.
    # Predict the derived satisfaction score from underlying customer metrics.
    feature_cols = [
        "xdr_sessions", "session_duration_ms", "total_traffic_bytes",
        "avg_tcp_retransmission", "avg_rtt", "avg_throughput",
        "engagement_cluster", "experience_cluster"
    ]
    X = scores[feature_cols]
    y = scores["satisfaction_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1, max_depth=12
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred))
    }
    return model, metrics

def final_satisfaction_clusters(scores):
    cols = ["engagement_score", "experience_score"]
    clustered, model, scaler, X = kmeans_clusters(scores, cols, 2)
    summary = clustered.groupby("cluster")[[
        "engagement_score", "experience_score", "satisfaction_score"
    ]].mean().reset_index()
    return clustered, summary, model, scaler
