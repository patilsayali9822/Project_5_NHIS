# Telecom User Analytics — TellCo

End-to-end implementation of the supplied telecom user analytics assignment.

## What it does
1. Cleans the 150k-session telecom dataset.
2. Handles missing values and IQR outliers using the assignment's mean-replacement rule.
3. Creates customer-level features.
4. Performs User Overview EDA, application analysis, deciles, correlation and PCA.
5. Performs User Engagement analysis with K-Means and elbow method.
6. Performs User Experience analysis with K-Means.
7. Builds engagement, experience and satisfaction scores.
8. Builds a satisfaction regression model.
9. Runs final K-Means with k=2.
10. Exports final scores to MySQL when configured.
11. Provides a Streamlit dashboard.
12. Includes tests, GitHub Actions and Docker.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Put the dataset at:

`data/raw/telcom_data.csv`

Then run:

```bash
python -m src.telecom_analytics.pipeline --input data/raw/telcom_data.csv
streamlit run dashboard/app.py
```

Or use the supplied dataset path directly:

```bash
python -m src.telecom_analytics.pipeline --input "telcom_data (2).xlsx - Sheet1.csv"
```

## MySQL

Set:

```bash
export MYSQL_URL="mysql+pymysql://user:password@localhost:3306/telco"
```

Windows PowerShell:

```powershell
$env:MYSQL_URL="mysql+pymysql://user:password@localhost:3306/telco"
```

Then rerun the pipeline.

## MLflow

Optional:

```bash
mlflow ui --backend-store-uri ./mlruns
```

The pipeline logs parameters, metrics and model artifacts when MLflow is installed.

## Docker

```bash
docker build -t telecom-user-analytics .
docker run --rm -p 8501:8501 telecom-user-analytics
```
