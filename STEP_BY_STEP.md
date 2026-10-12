# Step-by-step execution guide

## Step 1 — Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2 — Verify the dataset
The supplied file is copied to `data/raw/telcom_data.csv`.
Expected source shape: 150001 rows x 55 columns.

## Step 3 — Run the complete pipeline
```bash
python -m src.telecom_analytics.pipeline --input data/raw/telcom_data.csv
```

## Step 4 — Inspect outputs
Open `data/processed/`.
Important files:
- cleaned_data.csv
- missing_report_before.csv
- outlier_report.csv
- user_feature_store.csv
- top_10_handsets.csv
- top_3_manufacturers.csv
- top_5_handsets_by_top_manufacturer.csv
- descriptive_statistics.csv
- top_5_duration_deciles.csv
- application_correlation.csv
- pca_explained_variance.csv
- pca_loadings.csv
- engagement_elbow.csv
- engagement_clusters.csv
- experience_clusters.csv
- final_customer_scores.csv
- satisfaction_cluster_summary.csv
- regression_metrics.json

## Step 5 — Run tests
```bash
pytest -q
```

## Step 6 — Run dashboard
```bash
streamlit run dashboard/app.py
```

## Step 7 — MySQL
Create a database called `telco`, then set MYSQL_URL:
```text
mysql+pymysql://USERNAME:PASSWORD@localhost:3306/telco
```
Run the pipeline again. It will create `customer_satisfaction`.

## Step 8 — Docker
```bash
docker build -t telecom-user-analytics .
docker run --rm -p 8501:8501 telecom-user-analytics
```

## Step 9 — GitHub
```bash
git init
git add .
git commit -m "Initial telecom user analytics project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Step 10 — Presentation
Use the generated CSVs and figures to build the <=20 slide submission:
1. Title
2. Business objective
3. Dataset and variables
4. Data quality
5. Top handsets
6. Manufacturers
7. Application usage
8. Customer overview
9. Decile analysis
10. Correlation
11. PCA
12. Engagement
13. Engagement clusters
14. Experience
15. Experience clusters
16. Satisfaction
17. Regression
18. Final customer segments
19. Business recommendation
20. Limitations + references
