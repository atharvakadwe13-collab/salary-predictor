import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="Data Science Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────
st.title("💰 Data Science Salary Predictor")
st.markdown("**Built by Atharva Kadwe | BSc Computer Science @ BSBI Berlin**")
st.markdown("---")

# ── Load Data ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ds_salaries.csv")
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────
st.sidebar.header("🔍 Predict Your Salary")

experience = st.sidebar.selectbox(
    "Experience Level",
    ["EN - Entry Level", "MI - Mid Level", 
     "SE - Senior Level", "EX - Executive"]
)

employment = st.sidebar.selectbox(
    "Employment Type",
    ["FT - Full Time", "PT - Part Time", 
     "CT - Contract", "FL - Freelance"]
)

remote = st.sidebar.selectbox(
    "Remote Ratio",
    ["0 - On-site", "50 - Hybrid", "100 - Fully Remote"]
)

company_size = st.sidebar.selectbox(
    "Company Size",
    ["S - Small", "M - Medium", "L - Large"]
)

# ── Prepare Data ──────────────────────────────────────
df_model = df.copy()

le = LabelEncoder()
df_model["experience_level"] = le.fit_transform(df_model["experience_level"])
df_model["employment_type"] = le.fit_transform(df_model["employment_type"])
df_model["company_size"] = le.fit_transform(df_model["company_size"])

X = df_model[["experience_level", "employment_type", 
               "remote_ratio", "company_size"]]
y = df_model["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ── Prediction ────────────────────────────────────────
exp_map = {"EN - Entry Level": 0, "EX - Executive": 1,
           "MI - Mid Level": 2, "SE - Senior Level": 3}
emp_map = {"CT - Contract": 0, "FL - Freelance": 1,
           "FT - Full Time": 2, "PT - Part Time": 3}
size_map = {"L - Large": 0, "M - Medium": 1, "S - Small": 2}
remote_val = int(remote.split(" - ")[0])

input_data = np.array([[
    exp_map[experience],
    emp_map[employment],
    remote_val,
    size_map[company_size]
]])

prediction = model.predict(input_data)[0]

# ── Show Prediction ───────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.success(f"💵 Predicted Salary: **${prediction:,.0f} / year**")
st.sidebar.info(f"💶 In EUR: **€{prediction * 0.92:,.0f} / year**")

# ── Dashboard ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Total Records", len(df))

with col2:
    st.metric("💰 Avg Salary (USD)", 
              f"${df['salary_in_usd'].mean():,.0f}")

with col3:
    y_pred = model.predict(X_test)
    st.metric("🎯 Model Accuracy (R²)", 
              f"{r2_score(y_test, y_pred):.2%}")

st.markdown("---")

# ── Charts ────────────────────────────────────────────
col4, col5 = st.columns(2)

with col4:
    st.subheader("💼 Salary by Experience Level")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    exp_salary = df.groupby("experience_level")["salary_in_usd"].mean()
    exp_salary.plot(kind="bar", ax=ax1, color=["#1A5276", "#2980B9", 
                                                "#5DADE2", "#AED6F1"])
    ax1.set_xlabel("Experience Level")
    ax1.set_ylabel("Average Salary (USD)")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col5:
    st.subheader("🌍 Salary by Remote Ratio")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    remote_salary = df.groupby("remote_ratio")["salary_in_usd"].mean()
    remote_salary.plot(kind="bar", ax=ax2, color=["#1E8449", "#27AE60", 
                                                    "#82E0AA"])
    ax2.set_xlabel("Remote Ratio (%)")
    ax2.set_ylabel("Average Salary (USD)")
    plt.xticks(rotation=0)
    st.pyplot(fig2)

st.markdown("---")

# ── Data Table ────────────────────────────────────────
st.subheader("📋 Raw Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ by Atharva Kadwe | "
           "BSc Computer Science @ BSBI Berlin | "
           "GitHub: github.com/atharva-kadwe")