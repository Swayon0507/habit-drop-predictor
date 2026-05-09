import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Habit Drop Predictor", page_icon="🎯", layout="wide")

FEATURE_COLS = [
    "roll7_rate", "roll14_rate", "roll3_rate",
    "streak", "days_since_miss", "weekend_penalty",
    "habit_age", "is_weekend", "dayofweek", "mood",
    "habit_encoded"
]
THRESHOLD = 0.05

HABIT_HINTS = {
    "Sleep":           "✅ Completed if you slept 7+ hours (420+ minutes) that night",
    "Workout":         "✅ Completed if you had 20+ very active minutes that day",
    "Steps":           "✅ Completed if you walked 8,000+ steps that day",
    "Sedentary_Limit": "✅ Completed if sedentary time was 600 minutes or less",
}

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

# ── Manual Input Section ──────────────────────────────────────
st.divider()
st.subheader("🖊️ Try Your Own Habit Data")
st.markdown("Enter your own habit stats below to get an instant drop risk prediction.")

habit_input = st.selectbox(
    "Select a Habit",
    ["Sleep", "Workout", "Steps", "Sedentary_Limit"],
    format_func=lambda x: x.replace("_", " ")
)

st.info(f"**What counts as completing this habit?**  {HABIT_HINTS[habit_input]}")

st.markdown("---")
st.markdown("#### 📅 Your Recent Completion History")
st.markdown("*Count how many days you completed the habit in each period*")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**7-Day Completion Rate**")
    st.caption(f"Out of the last 7 days, how many did you complete {habit_input.replace('_',' ')}?")
    days7 = st.slider("Days completed out of last 7", 0, 7, 4, key="d7")
    roll7_input = days7 / 7

    st.markdown("**14-Day Completion Rate**")
    st.caption(f"Out of the last 14 days, how many did you complete {habit_input.replace('_',' ')}?")
    days14 = st.slider("Days completed out of last 14", 0, 14, 9, key="d14")
    roll14_input = days14 / 14

    st.markdown("**Last 3-Day Completion Rate**")
    st.caption("How many of the last 3 days did you complete it?")
    days3 = st.slider("Days completed out of last 3", 0, 3, 2, key="d3")
    roll3_input = days3 / 3 if days3 > 0 else 0.0

with col_b:
    st.markdown("**Current Streak**")
    st.caption("How many days IN A ROW have you completed it right now?")
    streak_input = st.number_input("Consecutive days", 0, 90, 3, key="streak")

    st.markdown("**Days Since Last Miss**")
    st.caption("How many days ago did you last SKIP this habit?")
    miss_input = st.number_input("Days ago", 0, 90, 2, key="miss")

    st.markdown("**Habit Age**")
    st.caption("How many days have you been tracking this habit in total?")
    age_input = st.number_input("Total days tracked", 1, 365, 30, key="age")

    st.markdown("**Your Mood Today**")
    st.caption("Rate how you feel today from 1 (very low) to 10 (excellent)")
    mood_input = st.slider("Mood score", 1, 10, 7, key="mood")

    st.markdown("**Weekend Performance**")
    st.caption("Are you worse at this habit on weekends vs weekdays?")
    weekend_opts = {
        "Same on weekdays and weekends": 0.0,
        "Slightly worse on weekends (~20%)": 0.2,
        "Much worse on weekends (~40%)": 0.4,
        "Better on weekends": -0.2,
    }
    weekend_sel   = st.selectbox("Weekend pattern", list(weekend_opts.keys()), key="wknd")
    weekend_input = weekend_opts[weekend_sel]

    st.markdown("**Day of Week**")
    dow_input   = st.selectbox("Today is", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], key="dow")
    is_we_input = 1 if dow_input in ["Sat","Sun"] else 0
    dow_num     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dow_input)

st.markdown("---")

st.markdown("#### 📋 Your Input Summary")
summary_cols = st.columns(4)
summary_cols[0].metric("7-Day Rate",      f"{round(roll7_input*100)}%  ({days7}/7 days)")
summary_cols[1].metric("Streak",          f"{streak_input} days in a row")
summary_cols[2].metric("Last 3-Day Rate", f"{round(roll3_input*100)}%  ({days3}/3 days)")
summary_cols[3].metric("Habit Age",       f"{age_input} days")

st.markdown("")
if st.button("🔮 Predict My Drop Risk", type="primary", use_container_width=True):
    try:
        habit_enc = le.transform([habit_input])[0]
    except Exception:
        habit_enc = 0

    input_df = pd.DataFrame([[
        roll7_input, roll14_input, roll3_input,
        streak_input, miss_input, weekend_input,
        age_input, is_we_input, dow_num,
        mood_input, habit_enc
    ]], columns=FEATURE_COLS)

    # ── Step 1: Get raw ML model probability ─────────────
    raw_prob = model.predict_proba(input_df)[0][1]

    # ── Step 2: Rule-based risk scoring ──────────────────
    # ML model is conservative due to class imbalance.
    # Domain knowledge rules act as a floor — hybrid approach.
    rule_score = 0.0

    # Rule 1: 7-day completion rate (strongest SHAP feature)
    if roll7_input <= 0.14:
        rule_score += 0.40
    elif roll7_input <= 0.28:
        rule_score += 0.25
    elif roll7_input <= 0.42:
        rule_score += 0.10

    # Rule 2: Broken or very short streak
    if streak_input == 0:
        rule_score += 0.20
    elif streak_input <= 2:
        rule_score += 0.08

    # Rule 3: Very low recent 3-day completion
    if roll3_input == 0.0:
        rule_score += 0.20
    elif roll3_input <= 0.33:
        rule_score += 0.10

    # Rule 4: Low mood
    if mood_input <= 3:
        rule_score += 0.10
    elif mood_input <= 5:
        rule_score += 0.05

    # Rule 5: Weekend performance penalty
    if weekend_input >= 0.4:
        rule_score += 0.10
    elif weekend_input >= 0.2:
        rule_score += 0.05

    # ── Step 3: Combine ML + rules (40% ML, 60% rules) ───
    risk     = min(0.40 * raw_prob + 0.60 * rule_score, 1.0)
    risk_pct = round(risk * 100, 1)

    # ── Step 4: Display result ────────────────────────────
    st.divider()
    if risk >= 0.6:
        st.error(f"🔴 HIGH RISK — {risk_pct}% chance of dropping **{habit_input.replace('_',' ')}** next week")
    elif risk >= 0.3:
        st.warning(f"🟠 MEDIUM RISK — {risk_pct}% chance of dropping **{habit_input.replace('_',' ')}** next week")
    else:
        st.success(f"🟢 LOW RISK — {risk_pct}% chance of dropping **{habit_input.replace('_',' ')}** next week")

    fake_row = pd.Series({
        "streak":          streak_input,
        "roll3_rate":      roll3_input,
        "weekend_penalty": weekend_input,
        "roll7_rate":      roll7_input,
        "habit_age":       age_input
    })
    suggestion = get_intervention(fake_row)
    if risk >= 0.3:
        st.info(f"💡 **Suggested Action:** {suggestion}")
    else:
        st.info("✅ Keep going! Your habit is stable. Maintain your current routine.")

st.divider()
st.caption("Habit Drop Predictor | Final Year Project | Powered by XGBoost + SHAP")