import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

ART = "artifacts"
st.set_page_config(layout="wide", page_title="Customer Personality Analyzer")

# Load data & artifacts
@st.cache_resource
def load_all():
    df = pd.read_csv("customers.csv")
    pre = joblib.load(os.path.join(ART, "preprocessor.joblib"))
    kmeans = joblib.load(os.path.join(ART, "kmeans.joblib"))
    personas = pd.read_csv(os.path.join(ART, "personas.csv"))
    rf = joblib.load(os.path.join(ART, "rf_churn.joblib"))
    xgb = joblib.load(os.path.join(ART, "xgb_churn.joblib"))
    fi = pd.read_csv(os.path.join(ART, "feature_importance.csv"))
    metrics = pd.read_csv(os.path.join(ART, "churn_metrics.csv"))
    return df, pre, kmeans, personas, rf, xgb, fi, metrics

df, preprocessor, kmeans, personas, rf_clf, xgb_clf, fi_df, metrics_df = load_all()

st.title("Customer Personality Analyzer")
st.markdown("Clusters customers, predicts churn, generates personas, and recommends marketing strategies.")

# Top row: summary metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Customers", f"{len(df):,}")
with col2:
    churn_rate = df["churn"].mean()
    st.metric("Overall Churn Rate", f"{churn_rate*100:.2f}%")
with col3:
    n_clusters = kmeans.n_clusters
    st.metric("Clusters", f"{n_clusters}")

st.write("---")

# Left: cluster explorer
st.subheader("Cluster explorer")
cluster_sel = st.selectbox("Choose cluster", sorted(df["cluster"].unique()))
cluster_df = df[df["cluster"]==cluster_sel]

c1, c2 = st.columns([1,2])
with c1:
    st.write(f"Cluster {cluster_sel} — {len(cluster_df)} customers")
    st.write("Persona summary:")
    per = personas[personas.cluster==cluster_sel].iloc[0]
    st.write(per.description)
    st.markdown("**Recommendations**")
    for r in per.recommendations:
        st.write("- " + r)

with c2:
    st.write("Key stats")
    stats = cluster_df[["age","avg_order_value","orders_per_month","ltv","recency_days","email_open_rate"]].describe().T[["mean","50%","std"]]
    stats = stats.rename(columns={"50%":"median"})
    st.dataframe(stats.style.format({"mean":"{:.2f}","median":"{:.2f}","std":"{:.2f}"}), height=250)

st.write("---")

# Churn model evaluation & top risk customers
st.subheader("Churn prediction")

st.write("Model metrics (AUC)")
st.table(metrics_df)

# Show top predicted risks (using RF by default)
st.write("Top customers at risk of churn (by RandomForest probability)")
X_all = df[preprocessor.feature_names_in_.tolist()] if hasattr(preprocessor, "feature_names_in_") else df  # fallback
X_proc = preprocessor.transform(df[["age","avg_order_value","orders_per_month","recency_days","tenure_months",
                                   "pref_electronics","pref_clothing","pref_home","pref_sports",
                                   "email_open_rate","app_sessions_per_week","ltv","gender","region"]])
probs = rf_clf.predict_proba(X_proc)[:,1]
df["churn_risk"] = probs
top_risk = df.sort_values("churn_risk", ascending=False).head(15)
st.dataframe(top_risk[["customer_id","age","region","ltv","orders_per_month","churn_risk"]].assign(churn_risk=lambda x: x.churn_risk.map("{:.3f}".format)), height=300)

st.write("---")

# Feature importance plot
st.subheader("Churn model feature importance (RandomForest)")
fig = px.bar(fi_df.sort_values("importance", ascending=True), x="importance", y="feature", orientation="h", height=400)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

# Interactive prediction for a single customer
st.subheader("Predict churn for a custom customer")
with st.form("custom_predict"):
    colA, colB, colC = st.columns(3)
    with colA:
        age = st.number_input("Age", 18, 90, 35)
        avg_order_value = st.number_input("Avg order value", 5.0, 10000.0, 80.0, step=1.0)
        orders_per_month = st.number_input("Orders per month", 0, 50, 1)
        recency_days = st.number_input("Recency (days since last purchase)", 0, 3650, 30)
    with colB:
        tenure_months = st.number_input("Tenure (months)", 0, 500, 12)
        email_open_rate = st.slider("Email open rate", 0.0, 1.0, 0.2)
        app_sessions_per_week = st.number_input("App sessions/week", 0, 100, 1)
    with colC:
        pref_electronics = st.slider("Pref electronics (0-1)", 0.0, 1.0, 0.2)
        pref_clothing = st.slider("Pref clothing (0-1)", 0.0, 1.0, 0.2)
        pref_home = st.slider("Pref home (0-1)", 0.0, 1.0, 0.2)
        pref_sports = st.slider("Pref sports (0-1)", 0.0, 1.0, 0.2)
        gender = st.selectbox("Gender", ["Male","Female","Other"])
        region = st.selectbox("Region", df["region"].unique().tolist())

    submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([{
            "age": age, "avg_order_value": avg_order_value, "orders_per_month": orders_per_month,
            "recency_days": recency_days, "tenure_months": tenure_months,
            "pref_electronics": pref_electronics, "pref_clothing": pref_clothing,
            "pref_home": pref_home, "pref_sports": pref_sports, "email_open_rate": email_open_rate,
            "app_sessions_per_week": app_sessions_per_week, "ltv": avg_order_value * orders_per_month * tenure_months,
            "gender": gender, "region": region
        }])
        Xrow = preprocessor.transform(row)
        p_rf = rf_clf.predict_proba(Xrow)[:,1][0]
        p_xgb = xgb_clf.predict_proba(Xrow)[:,1][0]
        st.write(f"RandomForest churn prob: **{p_rf:.3f}**")
        st.write(f"XGBoost churn prob: **{p_xgb:.3f}**")
        mean_p = (p_rf + p_xgb)/2
        st.metric("Average predicted churn risk", f"{mean_p:.2%}")

        # Suggest action based on risk and persona
        if mean_p > 0.5:
            st.warning("High churn risk — Recommend immediate win-back offers, churn survey, 1:1 outreach.")
        elif mean_p > 0.2:
            st.info("Medium risk — try targeted discounts, re-engagement campaigns.")
        else:
            st.success("Low risk — maintain with loyalty programs and upsell opportunities.")

st.write("---")
st.markdown("Generated personas (automated):")
for _, row in personas.iterrows():
    with st.expander(f"Cluster {int(row.cluster)} (size {int(row.size)})"):
        st.write(row.description)
        st.write("Recommendations:")
        for r in row.recommendations.strip("[]").split(","):
            st.write("-", r.strip().strip("'"))
