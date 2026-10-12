import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from .io import load_data, save_csv
from .cleaning import clean_dataset, missing_report
from .features import build_user_feature_store, engagement_features, experience_features
from .overview import (
    top_handsets, top_manufacturers, top_handsets_by_manufacturer,
    descriptive_statistics, decile_analysis, correlation, run_pca, save_overview_charts
)
from .clustering import elbow, kmeans_clusters, cluster_summary
from .satisfaction import build_scores, satisfaction_regression, final_satisfaction_clusters
from .database import export_mysql
from .config import DATA_PROCESSED, FIGURES, MODELS, USER_ID

def main(input_path):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("1/8 Loading data...")
    raw = load_data(input_path)
    print(f"Raw shape: {raw.shape}")

    print("2/8 Cleaning...")
    clean, outlier_report = clean_dataset(raw)
    save_csv(clean, DATA_PROCESSED/"cleaned_data.csv")
    save_csv(missing_report(raw).reset_index(names="column"),
             DATA_PROCESSED/"missing_report_before.csv")
    save_csv(outlier_report, DATA_PROCESSED/"outlier_report.csv")
    print(f"Cleaned shape: {clean.shape}")

    print("3/8 User overview...")
    user = build_user_feature_store(clean)
    save_csv(user, DATA_PROCESSED/"user_feature_store.csv")
    top_handsets(clean).to_csv(DATA_PROCESSED/"top_10_handsets.csv", index=False)
    top_manufacturers(clean).to_csv(DATA_PROCESSED/"top_3_manufacturers.csv", index=False)
    top_handsets_by_manufacturer(clean).to_csv(
        DATA_PROCESSED/"top_5_handsets_by_top_manufacturer.csv", index=False
    )
    descriptive_statistics(user).to_csv(DATA_PROCESSED/"descriptive_statistics.csv")
    decile_analysis(user).to_csv(DATA_PROCESSED/"top_5_duration_deciles.csv", index=False)
    correlation(user).to_csv(DATA_PROCESSED/"application_correlation.csv")
    pca_df, explained, loadings = run_pca(user)
    pca_df.insert(0, USER_ID, user[USER_ID].values)
    save_csv(pca_df, DATA_PROCESSED/"pca_scores.csv")
    explained.to_csv(DATA_PROCESSED/"pca_explained_variance.csv", index=False)
    loadings.to_csv(DATA_PROCESSED/"pca_loadings.csv")
    save_overview_charts(clean, user)

    print("4/8 Engagement clustering...")
    ecols = ["xdr_sessions", "session_duration_ms", "total_traffic_bytes"]
    _, _, _, X = kmeans_clusters(user, ecols, 3)
    elbow(X).to_csv(DATA_PROCESSED/"engagement_elbow.csv", index=False)
    engagement_clustered, e_model, e_scaler, _ = kmeans_clusters(user, ecols, 3)
    engagement_clustered.to_csv(DATA_PROCESSED/"engagement_clusters.csv", index=False)
    cluster_summary(engagement_clustered, ecols).to_csv(
        DATA_PROCESSED/"engagement_cluster_summary.csv"
    )

    print("5/8 Experience clustering...")
    xcols = ["avg_tcp_retransmission", "avg_rtt", "avg_throughput"]
    experience_clustered, x_model, x_scaler, _ = kmeans_clusters(user, xcols, 3)
    experience_clustered.to_csv(DATA_PROCESSED/"experience_clusters.csv", index=False)
    cluster_summary(experience_clustered, xcols).to_csv(
        DATA_PROCESSED/"experience_cluster_summary.csv"
    )

    print("6/8 Satisfaction...")
    scores, score_meta = build_scores(user)
    final_scores, final_summary, final_model, final_scaler = final_satisfaction_clusters(scores)
    final_scores.to_csv(DATA_PROCESSED/"final_customer_scores.csv", index=False)
    final_summary.to_csv(DATA_PROCESSED/"satisfaction_cluster_summary.csv", index=False)

    regression_input = user.merge(scores, on=USER_ID, how="inner")
    regression_model, regression_metrics = satisfaction_regression(regression_input)
    joblib.dump(regression_model, MODELS/"satisfaction_regression.joblib")
    joblib.dump(final_model, MODELS/"satisfaction_kmeans.joblib")
    joblib.dump(final_scaler, MODELS/"satisfaction_scaler.joblib")

    with open(DATA_PROCESSED/"regression_metrics.json", "w") as f:
        json.dump(regression_metrics, f, indent=2)
    with open(DATA_PROCESSED/"clustering_metadata.json", "w") as f:
        json.dump({
            "least_engaged_cluster": score_meta["least_engaged_cluster"],
            "worst_experience_cluster": score_meta["worst_experience_cluster"],
            "regression_metrics": regression_metrics
        }, f, indent=2)

    print("7/8 MySQL export...")
    print(export_mysql(final_scores))

    print("8/8 Complete.")
    print("Outputs are in data/processed/, reports/figures/, and models/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    main(args.input)
