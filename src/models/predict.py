import pandas as pd
import numpy as np
import joblib

FEATURE_COLS = [
    "roll7_rate", "roll14_rate", "roll3_rate",
    "streak", "days_since_miss", "weekend_penalty",
    "habit_age", "is_weekend", "dayofweek", "mood",
    "habit_encoded"
]

THRESHOLD = 0.3  # Lower threshold since drop events are rare

INTERVENTIONS = {
    "roll7_rate":      "Your 7-day completion rate is low. Try simplifying this habit for one week.",
    "streak":          "Your streak broke recently. Restart with a 2-day mini challenge.",
    "days_since_miss": "You missed this habit recently. Schedule a fixed reminder today.",
    "weekend_penalty": "You struggle on weekends. Plan this habit specifically for Saturday/Sunday.",
    "roll3_rate":      "You missed this habit 2-3 days in a row. Reduce difficulty immediately.",
    "habit_age":       "This is a newer habit still being formed. Add an extra reminder this week.",
}

def predict_risk(df, model, le):
    df["habit_encoded"] = le.transform(df["habit"])
    df["mood"] = df["mood"].fillna(5)
    X = df[FEATURE_COLS]
    proba = model.predict_proba(X)[:, 1]
    df["risk_score"]   = proba
    df["risk_percent"] = (proba * 100).round(1)
    df["will_drop"]    = (proba >= THRESHOLD).astype(int)
    return df

def get_intervention(row, model):
    reason = "roll7_rate"
    if row["streak"] == 0:
        reason = "streak"
    elif row["roll3_rate"] < 0.3:
        reason = "roll3_rate"
    elif row["weekend_penalty"] > 0.3:
        reason = "weekend_penalty"
    elif row["roll7_rate"] < 0.4:
        reason = "roll7_rate"
    return INTERVENTIONS[reason]

def generate_report(df, model):
    print("\n" + "="*60)
    print("  HABIT RISK REPORT")
    print("="*60)
    latest = df.sort_values("date").groupby("habit").last().reset_index()
    latest = predict_risk(latest, model, joblib.load("src/models/label_encoder.pkl"))
    latest = latest.sort_values("risk_score", ascending=False)
    print(f"\n  {'Habit':<20} {'Risk Score':>12} {'Status':>12}")
    print("  " + "-"*45)
    for _, row in latest.iterrows():
        status = "HIGH RISK" if row["risk_score"] >= THRESHOLD else "Stable"
        print(f"  {row['habit']:<20} {row['risk_percent']:>10.1f}%  {status:>10}")
        if row["risk_score"] >= THRESHOLD:
            print(f"  {'':20} Suggestion: {get_intervention(row, None)}")
    print("="*60)

if __name__ == "__main__":
    model = joblib.load("src/models/habit_drop_model.pkl")
    le    = joblib.load("src/models/label_encoder.pkl")
    df    = pd.read_csv("data/processed/real_user_processed.csv")
    generate_report(df, model)
