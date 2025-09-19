import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

# Demographics
ages = np.random.randint(18, 80, size=n)
genders = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48,0.48,0.04])
regions = np.random.choice(["North","South","East","West","Central"], size=n, p=[0.25,0.2,0.2,0.2,0.15])

# Behavioral / transactional
avg_order_value = np.random.exponential(scale=80, size=n) + 20  # positive skew
orders_per_month = np.random.poisson(lam=1.2, size=n) + (avg_order_value/400).astype(int)
orders_per_month = np.clip(orders_per_month, 0, 20)
recency_days = np.random.exponential(scale=60, size=n).astype(int)  # days since last purchase
tenure_months = np.random.randint(1, 120, size=n)

# Product affinity (one-hot-like probabilities)
pref_electronics = np.random.beta(2,5,size=n)
pref_clothing = np.random.beta(2,4,size=n)
pref_home = np.random.beta(1.5,3,size=n)
pref_sports = np.random.beta(1.2,4,size=n)

# Engagement
email_open_rate = np.clip(np.random.beta(2,5,size=n), 0, 1)
app_sessions_per_week = np.random.poisson(lam=1.5, size=n)

# Lifetime value (synthetic)
ltv = avg_order_value * orders_per_month * tenure_months * (1 + pref_electronics*0.3 + pref_clothing*0.1)
ltv = ltv + np.random.normal(0, 1000, size=n)
ltv = np.clip(ltv, 50, None)

# Synthetic churn flag — higher chance to churn if recency high, low orders, low engagement
churn_prob = (
    0.2*(recency_days/365) +
    0.3*(1/(1+orders_per_month)) +
    0.2*(1 - email_open_rate) +
    0.1*(1 - (ltv/ltv.max()))
)
churn_prob = np.clip(churn_prob, 0, 1)
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    "customer_id": np.arange(1, n+1),
    "age": ages,
    "gender": genders,
    "region": regions,
    "avg_order_value": np.round(avg_order_value, 2),
    "orders_per_month": orders_per_month,
    "recency_days": recency_days,
    "tenure_months": tenure_months,
    "pref_electronics": np.round(pref_electronics, 3),
    "pref_clothing": np.round(pref_clothing, 3),
    "pref_home": np.round(pref_home, 3),
    "pref_sports": np.round(pref_sports, 3),
    "email_open_rate": np.round(email_open_rate, 3),
    "app_sessions_per_week": app_sessions_per_week,
    "ltv": np.round(ltv, 2),
    "churn": churn
})

df.to_csv("customers.csv", index=False)
print("Saved customers.csv with", len(df), "rows.")
