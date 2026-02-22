import pandas as pd
import numpy as np
import os
import sys

THRESHOLDS = {
    "Sleep":        ("Sleep_Hours",          ">=", 7),
    "Workout":      ("Workout_Duration_Min",  ">=", 20),
    "Reading":      ("Reading_Min",          ">=", 10),
    "Screen_Limit": ("Screen_Time_Hours",     "<=", 3),
    "Budget":       ("Daily_Expense (RM)",    "<=", 50),
}
DROP_THRESHOLD_HIGH = 0.60
DROP_THRESHOLD_LOW  = 0.30
WINDOW_SIZE         = 7
OUTPUT_DIR          = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean(filepath):
    print(f"\n[1] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "Journaling (Y/N)" in df.columns:
        df["Journaling"] = df["Journaling (Y/N)"].str.strip().str.upper().map({"Y": 1, "N": 0})
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print(f"    Loaded {len(df)} rows | Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df

def binarize_habits(df):
    print("\n[2] Converting columns to binary completions")
    for habit_name, (col, op, threshold) in THRESHOLDS.items():
        if col not in df.columns:
            print(f"    WARNING: column '{col}' not found, skipping {habit_name}")
            continue
        if op == ">=":
            df[habit_name] = (df[col] >= threshold).astype(int)
        elif op == "<=":
            df[habit_name] = (df[col] <= threshold).astype(int)
        rate = df[habit_name].mean() * 100
        print(f"    {habit_name:15s} completion rate: {rate:.1f}%")
    if "Journaling" in df.columns:
        df["Journaling_Habit"] = df["Journaling"]
        print(f"    {'Journaling_Habit':15s} completion rate: {df['Journaling_Habit'].mean()*100:.1f}%")
    return df

def reshape_to_long(df, user_id="real_user_001"):
    print("\n[3] Reshaping to long format")
    habit_cols = list(THRESHOLDS.keys())
    if "Journaling_Habit" in df.columns:
        habit_cols.append("Journaling_Habit")
    rows = []
    for _, row in df.iterrows():
        for habit in habit_cols:
            if habit in df.columns:
                rows.append({
                    "user_id":   user_id,
                    "date":      row["Date"],
                    "habit":     habit,
                    "completed": int(row[habit]),
                    "mood":      row.get("Mood_Score (1-10)", np.nan),
                    "dayofweek": row["Date"].dayofweek,
                    "is_weekend": int(row["Date"].dayofweek >= 5),
                    "week_num":  row["Date"].isocalendar().week,
                    "day_num":   (row["Date"] - df["Date"].min()).days + 1,
                })
    long_df = pd.DataFrame(rows).sort_values(["user_id", "habit", "date"]).reset_index(drop=True)
    print(f"    Shape: {long_df.shape} | Habits: {long_df['habit'].unique().tolist()}")
    return long_df

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
        grp["days_since_miss"] = days_since_miss(grp["completed"].shift(1).fillna(1).tolist())
        weekday_rate = grp[grp["is_weekend"]==0]["completed"].mean()
        weekend_rate = grp[grp["is_weekend"]==1]["completed"].mean()
        grp["weekend_penalty"] = weekday_rate - weekend_rate
        grp["habit_age"] = grp["day_num"]
        feature_rows.append(grp)
    feat_df = pd.concat(feature_rows).reset_index(drop=True)
    fill_cols = ["roll7_rate","roll14_rate","roll3_rate","streak","days_since_miss"]
    feat_df[fill_cols] = feat_df[fill_cols].fillna(0)
    print("    Features added: roll7_rate, roll14_rate, roll3_rate, streak, days_since_miss, weekend_penalty, habit_age")
    return feat_df

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
    total = len(labeled_df)
    print(f"    Total rows: {total} | Drop events: {drop_count} ({drop_count/total*100:.1f}%) | Stable: {total-drop_count}")
    return labeled_df

def generate_synthetic_users(n_users=30, n_days=90):
    print(f"\n[6] Generating synthetic data for {n_users} users")
    np.random.seed(42)
    habits = ["Sleep","Workout","Reading","Screen_Limit","Budget","Journaling_Habit"]
    archetypes = {
        "consistent":    {"base_rate": 0.80, "decay": 0.002,  "weekend_drop": 0.05},
        "early_dropout": {"base_rate": 0.85, "decay": 0.015,  "weekend_drop": 0.15},
        "weekend_hero":  {"base_rate": 0.50, "decay": 0.001,  "weekend_drop": -0.20},
        "inconsistent":  {"base_rate": 0.55, "decay": 0.005,  "weekend_drop": 0.20},
        "late_bloomer":  {"base_rate": 0.30, "decay": -0.008, "weekend_drop": 0.10},
    }
    all_rows = []
    start_date = pd.Timestamp("2024-01-01")
    for u in range(n_users):
        user_id   = f"synthetic_user_{u+1:03d}"
        archetype = np.random.choice(list(archetypes.keys()))
        profile   = archetypes[archetype]
        for habit in habits:
            habit_base = np.clip(profile["base_rate"] + np.random.uniform(-0.15, 0.15), 0.1, 0.95)
            for day in range(n_days):
                date = start_date + pd.Timedelta(days=day)
                is_weekend = int(date.dayofweek >= 5)
                p = np.clip(habit_base - profile["decay"]*day - is_weekend*profile["weekend_drop"] + np.random.normal(0,0.05), 0.0, 1.0)
                all_rows.append({
                    "user_id": user_id, "archetype": archetype, "date": date,
                    "habit": habit, "completed": int(np.random.random() < p),
                    "mood": np.random.randint(4,11), "dayofweek": date.dayofweek,
                    "is_weekend": is_weekend, "week_num": date.isocalendar().week,
                    "day_num": day+1,
                })
    synthetic_df = pd.DataFrame(all_rows)
    print(f"    Synthetic rows: {len(synthetic_df)}")
    return synthetic_df

def run_pipeline(csv_path):
    print("="*60)
    print("  TINY HABIT DROP PREDICTOR - PREPROCESSING PIPELINE")
    print("="*60)
    raw        = load_and_clean(csv_path)
    binary     = binarize_habits(raw)
    long_df    = reshape_to_long(binary, user_id="real_user_001")
    feat_df    = engineer_features(long_df)
    labeled_df = create_target(feat_df)
    synthetic_long    = generate_synthetic_users(n_users=30, n_days=90)
    synthetic_feat    = engineer_features(synthetic_long)
    synthetic_labeled = create_target(synthetic_feat)
    combined = pd.concat([labeled_df, synthetic_labeled], ignore_index=True)
    labeled_df.to_csv(f"{OUTPUT_DIR}/real_user_processed.csv", index=False)
    synthetic_labeled.to_csv(f"{OUTPUT_DIR}/synthetic_users_processed.csv", index=False)
    combined.to_csv(f"{OUTPUT_DIR}/combined_dataset.csv", index=False)
    print(f"\n  Saved 3 files to {OUTPUT_DIR}/")
    print(f"  Combined shape: {combined.shape}")
    print(f"  Drop events: {combined['will_drop'].sum()} / {len(combined)}")
    print("="*60)
    print("  DONE - Next step: run model training script")
    print("="*60)
    return combined

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/90_day_habit_tracker.csv"
    run_pipeline(csv_file)
