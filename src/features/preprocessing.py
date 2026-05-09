"""
=============================================================
 Tiny Habit Drop Predictor — FitBit Preprocessing Pipeline
=============================================================
Dataset : FitBit Fitness Tracker Data (Kaggle - arashnic)
Steps   :
  1. Load dailyActivity + sleepDay CSV files
  2. Merge into one daily record per user per date
  3. Binarize into habit completions
  4. Reshape wide → long format
  5. Feature engineering (streaks, rolling rates, etc.)
  6. Create target variable (will_drop_next_7_days)
  7. Save processed datasets
=============================================================
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Configuration ────────────────────────────────────────
THRESHOLDS = {
    "Sleep":         ("TotalMinutesAsleep", ">=", 420),  # 7 hours
    "Workout":       ("VeryActiveMinutes",  ">=", 20),
    "Steps":         ("TotalSteps",         ">=", 8000),
    "Sedentary_Limit":("SedentaryMinutes",  "<=", 600),
}
DROP_THRESHOLD_HIGH = 0.60
DROP_THRESHOLD_LOW  = 0.30
WINDOW_SIZE         = 7
OUTPUT_DIR          = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load & Merge ─────────────────────────────────
def load_fitbit_data(data_dir="data/raw/fitbit"):
    print("\n[1] Loading FitBit data files")

    activity = pd.read_csv(os.path.join(data_dir, "dailyActivity_merged.csv"))
    sleep    = pd.read_csv(os.path.join(data_dir, "sleepDay_merged.csv"))

    # Parse dates
    activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"])
    sleep["SleepDay"]        = pd.to_datetime(sleep["SleepDay"]).dt.normalize()

    # Rename for merging
    activity = activity.rename(columns={"ActivityDate": "Date"})
    sleep    = sleep.rename(columns={"SleepDay": "Date"})

    # Keep only useful sleep columns
    sleep = sleep[["Id", "Date", "TotalMinutesAsleep", "TotalTimeInBed"]]

    # Merge activity + sleep on user ID and date
    df = pd.merge(activity, sleep, on=["Id", "Date"], how="left")

    # Fill missing sleep with 0 (user didn't log sleep that day)
    df["TotalMinutesAsleep"] = df["TotalMinutesAsleep"].fillna(0)
    df["TotalTimeInBed"]     = df["TotalTimeInBed"].fillna(0)

    # Rename Id to user_id
    df = df.rename(columns={"Id": "user_id"})

    print(f"    Users: {df['user_id'].nunique()}")
    print(f"    Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"    Total rows: {len(df)}")
    return df

# ── Step 2: Binarize ─────────────────────────────────────
def binarize_habits(df):
    print("\n[2] Binarizing habits")
    for habit, (col, op, threshold) in THRESHOLDS.items():
        if col not in df.columns:
            print(f"    WARNING: {col} not found, skipping {habit}")
            continue
        if op == ">=":
            df[habit] = (df[col] >= threshold).astype(int)
        elif op == "<=":
            df[habit] = (df[col] <= threshold).astype(int)
        rate = df[habit].mean() * 100
        print(f"    {habit:20s} → {rate:.1f}% completion rate")
    return df

# ── Step 3: Reshape Wide → Long ──────────────────────────
def reshape_to_long(df):
    print("\n[3] Reshaping to long format")
    habit_cols = list(THRESHOLDS.keys())
    rows = []
    for _, row in df.iterrows():
        for habit in habit_cols:
            if habit not in df.columns:
                continue
            rows.append({
                "user_id":    str(row["user_id"]),
                "date":       row["Date"],
                "habit":      habit,
                "completed":  int(row[habit]),
                "mood":       5,  # FitBit has no mood — use neutral default
                "dayofweek":  row["Date"].dayofweek,
                "is_weekend": int(row["Date"].dayofweek >= 5),
                "week_num":   row["Date"].isocalendar().week,
                "day_num":    1,
            })
    long_df = pd.DataFrame(rows)

    # Add day_num per user per habit
    long_df = long_df.sort_values(["user_id", "habit", "date"]).reset_index(drop=True)
    long_df["day_num"] = long_df.groupby(["user_id", "habit"]).cumcount() + 1

    print(f"    Shape: {long_df.shape}")
    print(f"    Habits: {long_df['habit'].unique().tolist()}")
    print(f"    Users:  {long_df['user_id'].nunique()}")
    return long_df

# ── Step 4: Feature Engineering ──────────────────────────
def compute_streak(series):
    streaks, count = [], 0
    for val in series:
        count = count + 1 if val == 1 else 0
        streaks.append(count)
    return pd.Series(streaks, index=series.index)

def engineer_features(long_df):
    print("\n[4] Engineering features")
    feature_rows = []
    for (user, habit), grp in long_df.groupby(["user_id", "habit"]):
        grp = grp.sort_values("date").reset_index(drop=True)

        grp["roll7_rate"]  = grp["completed"].shift(1).rolling(7,  min_periods=1).mean()
        grp["roll14_rate"] = grp["completed"].shift(1).rolling(14, min_periods=1).mean()
        grp["roll3_rate"]  = grp["completed"].shift(1).rolling(3,  min_periods=1).mean()
        grp["streak"]      = compute_streak(grp["completed"].shift(1).fillna(0))

        def days_since_miss(series):
            result, count = [], 0
            for v in series:
                result.append(count)
                count = 0 if v == 0 else count + 1
            return result
        grp["days_since_miss"] = days_since_miss(
            grp["completed"].shift(1).fillna(1).tolist()
        )

        weekday_rate = grp[grp["is_weekend"]==0]["completed"].mean()
        weekend_rate = grp[grp["is_weekend"]==1]["completed"].mean()
        grp["weekend_penalty"] = weekday_rate - weekend_rate

        grp["habit_age"] = grp["day_num"]
        feature_rows.append(grp)

    feat_df = pd.concat(feature_rows).reset_index(drop=True)
    fill_cols = ["roll7_rate","roll14_rate","roll3_rate","streak","days_since_miss"]
    feat_df[fill_cols] = feat_df[fill_cols].fillna(0)
    print("    Features added: roll7_rate, roll14_rate, roll3_rate,")
    print("                    streak, days_since_miss, weekend_penalty, habit_age")
    return feat_df

# ── Step 5: Target Variable ───────────────────────────────
def create_target(feat_df):
    print("\n[5] Creating target variable: will_drop")
    labeled_rows = []
    for (user, habit), grp in feat_df.groupby(["user_id", "habit"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        n, labels = len(grp), []
        for i in range(n):
            past_start = max(0, i - WINDOW_SIZE)
            past_rate  = grp.loc[past_start:i-1, "completed"].mean() if i > 0 else 0
            future_end  = min(n-1, i + WINDOW_SIZE)
            future_rate = grp.loc[i+1:future_end, "completed"].mean() if i < n-1 else np.nan
            if np.isnan(future_rate):
                labels.append(np.nan)
            elif past_rate >= DROP_THRESHOLD_HIGH and future_rate <= DROP_THRESHOLD_LOW:
                labels.append(1)
            else:
                labels.append(0)
        grp["will_drop"] = labels
        labeled_rows.append(grp)

    labeled_df = pd.concat(labeled_rows).reset_index(drop=True)
    labeled_df = labeled_df.dropna(subset=["will_drop"]).copy()
    labeled_df["will_drop"] = labeled_df["will_drop"].astype(int)

    drop_count = labeled_df["will_drop"].sum()
    total      = len(labeled_df)
    print(f"    Total rows  : {total}")
    print(f"    Drop events : {drop_count} ({drop_count/total*100:.1f}%)")
    print(f"    Stable      : {total - drop_count}")
    return labeled_df

# ── Step 6: Synthetic Users ───────────────────────────────
def generate_synthetic_users(n_users=50, n_days=31):
    print(f"\n[6] Generating {n_users} synthetic users × {n_days} days")
    np.random.seed(42)
    habits = list(THRESHOLDS.keys())
    archetypes = {
        "consistent":    {"base_rate": 0.80, "decay": 0.002,  "weekend_drop": 0.05},
        "early_dropout": {"base_rate": 0.85, "decay": 0.020,  "weekend_drop": 0.15},
        "weekend_hero":  {"base_rate": 0.50, "decay": 0.001,  "weekend_drop": -0.20},
        "inconsistent":  {"base_rate": 0.55, "decay": 0.005,  "weekend_drop": 0.20},
        "late_bloomer":  {"base_rate": 0.30, "decay": -0.010, "weekend_drop": 0.10},
    }
    all_rows = []
    start_date = pd.Timestamp("2016-03-12")
    for u in range(n_users):
        user_id   = f"synthetic_{u+1:03d}"
        archetype = np.random.choice(list(archetypes.keys()))
        profile   = archetypes[archetype]
        for habit in habits:
            habit_base = np.clip(
                profile["base_rate"] + np.random.uniform(-0.15, 0.15), 0.1, 0.95
            )
            for day in range(n_days):
                date = start_date + pd.Timedelta(days=day)
                is_weekend = int(date.dayofweek >= 5)
                p = np.clip(
                    habit_base
                    - profile["decay"] * day
                    - is_weekend * profile["weekend_drop"]
                    + np.random.normal(0, 0.05),
                    0.0, 1.0
                )
                all_rows.append({
                    "user_id":    user_id,
                    "date":       date,
                    "habit":      habit,
                    "completed":  int(np.random.random() < p),
                    "mood":       np.random.randint(4, 11),
                    "dayofweek":  date.dayofweek,
                    "is_weekend": is_weekend,
                    "week_num":   date.isocalendar().week,
                    "day_num":    day + 1,
                })
    syn_df = pd.DataFrame(all_rows)
    print(f"    Synthetic rows: {len(syn_df)}")
    return syn_df

# ── Main Pipeline ─────────────────────────────────────────
def run_pipeline(data_dir="data/raw/fitbit"):
    print("=" * 60)
    print("  HABIT DROP PREDICTOR — FITBIT PREPROCESSING PIPELINE")
    print("=" * 60)

    raw        = load_fitbit_data(data_dir)
    binary     = binarize_habits(raw)
    long_df    = reshape_to_long(binary)
    feat_df    = engineer_features(long_df)
    labeled_df = create_target(feat_df)

    syn_long    = generate_synthetic_users(n_users=50, n_days=31)
    syn_feat    = engineer_features(syn_long)
    syn_labeled = create_target(syn_feat)

    combined = pd.concat([labeled_df, syn_labeled], ignore_index=True)

    labeled_df.to_csv(f"{OUTPUT_DIR}/real_user_processed.csv",      index=False)
    syn_labeled.to_csv(f"{OUTPUT_DIR}/synthetic_users_processed.csv", index=False)
    combined.to_csv(f"{OUTPUT_DIR}/combined_dataset.csv",            index=False)

    print(f"\n  Saved 3 files to {OUTPUT_DIR}/")
    print(f"  Real user rows     : {len(labeled_df)}")
    print(f"  Synthetic rows     : {len(syn_labeled)}")
    print(f"  Combined rows      : {len(combined)}")
    print(f"  Total drop events  : {combined['will_drop'].sum()}")
    print("=" * 60)
    print("  DONE — Next: python src/models/train.py")
    print("=" * 60)
    return combined

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/fitbit"
    run_pipeline(data_dir)
