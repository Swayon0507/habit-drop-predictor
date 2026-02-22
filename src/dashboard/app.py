import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Habit Drop Predictor", page_icon="🎯", layout="wide")

FEATURE_COLS = [
    "roll7_rate", "roll14_rate", "roll3_rate",
    "streak", "days_since_miss", "weekend_penalty",
    "habit_age", "is_weekend", "dayofweek", "mood",
    "habit_encoded"
]
THRESHOLD = 0.3

INTERVENTIONS = {
    "streak":          "Your streak broke recently. Restart with a 2-day mini challenge.",
    "roll3_rate":      "You missed this habit 2-3 days in a row. Reduce difficulty immediately.",
    "weekend_penalty": "You struggle on weekends. Plan this habit specifically for Saturday/Sunday.",
    "roll7_rate":      "Your 7-day completion rate is low. Simplify this habit for one week.",
    "habit_age":       "This is a newer habit. Add an extra reminder this week.",
}

@st.cache_resource
def load_model():
    model = joblib.load("src/models/habit_drop_model.pkl")
    le    = joblib.load("src/models/label_encoder.pkl")
    return model, le

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/real_user_processed.csv")

def get_intervention(row):
    if row["streak"] == 0:
        return INTERVENTIONS["streak"]
    elif row["roll3_rate"] < 0.3:
        return INTERVENTIONS["roll3_rate"]
    elif row["weekend_penalty"] > 0.3:
        return INTERVENTIONS["weekend_penalty"]
    elif row["roll7_rate"] < 0.4:
        return INTERVENTIONS["roll7_rate"]
    return INTERVENTIONS["habit_age"]

def predict_habits(df, model, le):
    latest = df.sort_values("date").groupby("habit").last().reset_index()
    latest["habit_encoded"] = le.transform(latest["habit"])
    latest["mood"] = latest["mood"].fillna(5)
    X = latest[FEATURE_COLS]
    proba = model.predict_proba(X)[:, 1]
    latest["risk_score"]   = proba
    latest["risk_percent"] = (proba * 100).round(1)
    latest["at_risk"]      = proba >= THRESHOLD
    latest["intervention"] = latest.apply(get_intervention, axis=1)
    return latest.sort_values("risk_score", ascending=False)

def risk_color(score):
    if score >= 0.6:   return "#FF4B4B"
    elif score >= 0.3: return "#FFA500"
    else:              return "#00CC96"

def risk_label(score):
    if score >= 0.6:   return "🔴 HIGH RISK"
    elif score >= 0.3: return "🟠 MEDIUM RISK"
    else:              return "🟢 STABLE"

model, le = load_model()
df        = load_data()
results   = predict_habits(df, model, le)

st.title("🎯 Habit Drop Predictor")
st.markdown("*Proactive habit risk detection with personalized interventions*")
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Habits Tracked", len(results))
with col2:
    at_risk = results["at_risk"].sum()
    st.metric("Habits At Risk", int(at_risk), delta=f"{at_risk} need attention", delta_color="inverse")
with col3:
    avg_risk = results["risk_percent"].mean()
    st.metric("Average Risk Score", f"{avg_risk:.1f}%")
with col4:
    best = results.iloc[-1]["habit"].replace("_", " ")
    st.metric("Most Consistent Habit", best)

st.divider()

st.subheader("📊 Habit Risk Overview")
colors = [risk_color(s) for s in results["risk_score"]]
fig = go.Figure(go.Bar(
    x=results["habit"].str.replace("_", " "),
    y=results["risk_percent"],
    marker_color=colors,
    text=[f"{v:.1f}%" for v in results["risk_percent"]],
    textposition="outside"
))
fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
fig.add_hline(y=60, line_dash="dash", line_color="red",    annotation_text="High Risk Threshold")
fig.update_layout(
    xaxis_title="Habit", yaxis_title="Drop Risk (%)",
    yaxis_range=[0, 100], height=400,
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Habit Details & Interventions")
for _, row in results.iterrows():
    habit_name = row["habit"].replace("_", " ")
    label      = risk_label(row["risk_score"])
    with st.expander(f"{label}  |  {habit_name}  —  Risk: {row['risk_percent']:.1f}%"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("7-Day Rate",     f"{row['roll7_rate']*100:.0f}%")
        c2.metric("Current Streak", f"{int(row['streak'])} days")
        c3.metric("3-Day Rate",     f"{row['roll3_rate']*100:.0f}%")
        c4.metric("Habit Age",      f"{int(row['habit_age'])} days")
        if row["at_risk"]:
            st.warning(f"⚠️ **Intervention Suggested:** {row['intervention']}")
        else:
            st.success("✅ This habit is on track. Keep going!")

st.subheader("📈 Completion Trends Over Time")
selected = st.selectbox(
    "Select habit to view trend:",
    results["habit"].str.replace("_", " ").tolist()
)
selected_raw = selected.replace(" ", "_")
habit_df = df[df["habit"] == selected_raw].sort_values("date")
habit_df["roll7"] = habit_df["completed"].rolling(7, min_periods=1).mean() * 100

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=habit_df["date"], y=habit_df["completed"] * 100,
    name="Daily Completion", marker_color="lightblue", opacity=0.5
))
fig2.add_trace(go.Scatter(
    x=habit_df["date"], y=habit_df["roll7"],
    name="7-Day Average", line=dict(color="royalblue", width=2)
))
fig2.update_layout(
    xaxis_title="Date", yaxis_title="Completion %",
    yaxis_range=[0, 110], height=350,
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Habit Drop Predictor | Final Year Project | Powered by XGBoost + SHAP")
