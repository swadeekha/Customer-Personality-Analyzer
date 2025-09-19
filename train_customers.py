import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("customers.csv")
print("Loaded", df.shape)

# Features we will use
num_features = [
    "age", "avg_order_value", "orders_per_month", "recency_days",
    "tenure_months", "pref_electronics", "pref_clothing",
    "pref_home", "pref_sports", "email_open_rate", "app_sessions_per_week", "ltv"
]
cat_features = ["gender", "region"]

X = df[num_features + cat_features].copy()
y = df["churn"].values

# Preprocessing
num_pipe = Pipeline([
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", cat_pipe, cat_features)
])

X_proc = preprocessor.fit_transform(X)
print("Processed feature matrix shape:", X_proc.shape)

# 1) Clustering: choose k via silhouette (try 2..8)
best_k = 3
best_score = -1
scores = {}
for k in range(2,9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_proc)
    s = silhouette_score(X_proc, labels)
    scores[k] = s
    if s > best_score:
        best_k = k
        best_score = s

print("Silhouette scores by k:", scores)
print("Chosen k:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_proc)
df["cluster"] = cluster_labels

# Save clustering artifacts
joblib.dump(preprocessor, os.path.join(OUT, "preprocessor.joblib"))
joblib.dump(kmeans, os.path.join(OUT, "kmeans.joblib"))
print("Saved preprocessor and kmeans.")

# Cluster summaries -> personas
personas = []
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    size = len(sub)
    mean_vals = sub[num_features].mean()
    top_pref = mean_vals[["pref_electronics","pref_clothing","pref_home","pref_sports"]].idxmax()
    persona = {
        "cluster": int(c),
        "size": int(size),
        "mean_age": float(mean_vals["age"]),
        "mean_ltv": float(mean_vals["ltv"]),
        "avg_orders_per_month": float(mean_vals["orders_per_month"]),
        "mean_recency_days": float(mean_vals["recency_days"]),
        "top_pref": top_pref,
        "churn_rate": float(sub["churn"].mean())
    }
    # auto-generate a short description
    desc = f"Cluster {c}: {size} customers. Avg age {persona['mean_age']:.0f}. "
    desc += f"Top interest: {persona['top_pref'].replace('pref_','').title()}. "
    desc += f"Avg orders/mo {persona['avg_orders_per_month']:.2f}, LTV ${persona['mean_ltv']:.0f}. "
    desc += f"Churn rate: {persona['churn_rate']*100:.1f}%."
    persona["description"] = desc

    # Simple recommendation rules
    recs = []
    if persona["churn_rate"] > 0.25:
        recs.append("High churn: prioritize win-back campaigns and personalized offers")
    else:
        recs.append("Stable: loyalty/program offers to increase spend")
    if persona["avg_orders_per_month"] < 0.5:
        recs.append("Target with frequent small-purchase promotions and reminders")
    else:
        recs.append("Upsell bundles and cross-sell based on top categories")
    if persona["top_pref"] == "pref_electronics":
        recs.append("Recommend tech bundles, financing options")
    elif persona["top_pref"] == "pref_clothing":
        recs.append("Use style inspiration and seasonal discounts")
    persona["recommendations"] = recs

    personas.append(persona)

pd.DataFrame(personas).to_csv(os.path.join(OUT, "personas.csv"), index=False)
print("Saved personas.csv")

# 2) Churn prediction: train/test split
X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
xgb_clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=4)

# Evaluate with cross-val and test
rf_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
xgb_scores = cross_val_score(xgb_clf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)

rf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb_clf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]
y_proba_xgb = xgb_clf.predict_proba(X_test)[:,1]

rf_auc = roc_auc_score(y_test, y_proba_rf)
xgb_auc = roc_auc_score(y_test, y_proba_xgb)
print("RF CV AUC:", rf_scores.mean(), "Test AUC:", rf_auc)
print("XGB CV AUC:", xgb_scores.mean(), "Test AUC:", xgb_auc)

# Save models and some metrics
joblib.dump(rf, os.path.join(OUT, "rf_churn.joblib"))
joblib.dump(xgb_clf, os.path.join(OUT, "xgb_churn.joblib"))
pd.DataFrame({
    "model":["rf","xgb"],
    "cv_auc":[rf_scores.mean(), xgb_scores.mean()],
    "test_auc":[rf_auc, xgb_auc]
}).to_csv(os.path.join(OUT,"churn_metrics.csv"), index=False)

# Save sample of top features importance (using RF)
importances = rf.feature_importances_
# Need to map feature names after preprocessing:
# get transformed cat names
ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
cat_names = ohe.get_feature_names_out(cat_features)
feature_names = num_features + list(cat_names)
fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
fi_df = pd.DataFrame(fi, columns=["feature","importance"])
fi_df.to_csv(os.path.join(OUT,"feature_importance.csv"), index=False)
print("Saved churn models, metrics, and feature importance to artifacts/")

print("Training pipeline complete.")
