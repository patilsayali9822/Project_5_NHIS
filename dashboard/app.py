import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="TellCo User Analytics", layout="wide")
st.title("📱 TellCo Telecom User Analytics")
st.caption("User Overview • Engagement • Experience • Satisfaction")

scores = pd.read_csv(DATA/"final_customer_scores.csv")
users = pd.read_csv(DATA/"user_feature_store.csv")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Engagement", "Experience", "Satisfaction"
])

with tab1:
    st.subheader("User Overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Customers", f"{len(users):,}")
    c2.metric("Sessions", f"{users['xdr_sessions'].sum():,.0f}")
    c3.metric("Total Traffic (GB)", f"{users['total_traffic_bytes'].sum()/1e9:,.2f}")
    apps = [c for c in users.columns if c.endswith("_total")]
    app_df = users[apps].sum().sort_values(ascending=False).reset_index()
    app_df.columns = ["application","bytes"]
    st.plotly_chart(px.bar(app_df, x="application", y="bytes",
                           title="Application Traffic"), use_container_width=True)

with tab2:
    st.subheader("Engagement")
    counts = scores["engagement_cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster","customers"]
    st.plotly_chart(px.bar(counts, x="cluster", y="customers",
                           title="Engagement Cluster Sizes"), use_container_width=True)
    st.dataframe(scores.nlargest(10, "engagement_score_100")[
        ["MSISDN/Number","engagement_score_100","engagement_cluster"]
    ])

with tab3:
    st.subheader("Experience")
    counts = scores["experience_cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster","customers"]
    st.plotly_chart(px.bar(counts, x="cluster", y="customers",
                           title="Experience Cluster Sizes"), use_container_width=True)
    st.dataframe(scores[["MSISDN/Number","experience_score_100",
                         "experience_cluster"]].head(20))

with tab4:
    st.subheader("Customer Satisfaction")
    st.metric("Average Satisfaction", f"{scores['satisfaction_score_100'].mean():.2f}/100")
    st.plotly_chart(px.histogram(scores, x="satisfaction_score_100",
                                 nbins=40, title="Satisfaction Distribution"),
                    use_container_width=True)
    st.dataframe(scores.nlargest(10, "satisfaction_score_100")[
        ["MSISDN/Number","engagement_score_100",
         "experience_score_100","satisfaction_score_100"]
    ])
