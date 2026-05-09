# 🎯 Habit Drop Predictor

> **Proactive Habit Drop Detection and Intervention System Using Behavioral Time-Series Analysis**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red?logo=streamlit)](https://streamlit.io)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9194-brightgreen)](https://github.com/Swayon0507/habit-drop-predictor)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

A machine learning system that predicts which micro-habits a person is likely
to **drop in the upcoming week** and suggests **personalized corrective interventions**
before the dropout occurs. Built using real FitBit tracker data from 33 users,
time-series feature engineering, and a calibrated XGBoost classifier with SMOTE
oversampling for class imbalance handling.

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **AUC-ROC Score** | **0.9194** |
| **Brier Score** | **0.0298** |
| Real Users (FitBit) | 33 users |
| Synthetic Users | 50 users |
| Combined Dataset | 9,640 rows |
| Drop Events | 313 (3.2%) |
| Train / Test Split | 80% / 20% (GroupShuffleSplit) |
| Habits Tracked | Sleep, Workout, Steps, Sedentary Limit |

---

## ✨ Key Features

- 🔮 **Predicts habit drop risk** 7 days before it happens
- 🧠 **XGBoost + SMOTE** — handles severe class imbalance (27.9:1 ratio)
- 📊 **SHAP Explainability** — shows WHY each habit is at risk
- 💡 **Intervention Engine** — diagnoses cause and prescribes specific action
- 🖥️ **Live Streamlit Dashboard** — interactive risk visualization
- 🖊️ **Manual Input** — any user can enter their own data for instant prediction
- 🔄 **Probability Calibration** — isotonic regression ensures trustworthy risk scores

---

## 📁 Project Structure

```
habit-drop-predictor/
├── data/
│   ├── raw/
│   │   └── fitbit/                    ← FitBit Fitness Tracker CSVs
│   │       ├── dailyActivity_merged.csv
│   │       ├── sleepDay_merged.csv
│   │       ├── dailyCalories_merged.csv
│   │       ├── dailyIntensities_merged.csv
│   │       └── dailySteps_merged.csv
│   └── processed/
│       ├── real_user_processed.csv    ← FitBit users processed
│       ├── synthetic_users_processed.csv
│       └── combined_dataset.csv       ← Final training data
├── notebooks/
│   └── 01_eda.ipynb                   ← Exploratory data analysis
├── src/
│   ├── features/
│   │   └── preprocessing.py           ← FitBit data pipeline
│   ├── models/
│   │   ├── train.py                   ← XGBoost + SMOTE + calibration
│   │   └── predict.py                 ← Risk scoring + interventions
│   └── dashboard/
│       └── app.py                     ← Streamlit dashboard
├── reports/
│   └── figures/                       ← SHAP plots and visualizations
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

**Primary:** [FitBit Fitness Tracker Data](https://www.kaggle.com/datasets/arashnic/fitbit) — Kaggle (CC0 Public Domain)

| Property | Value |
|---|---|
| Users | 33 real Fitbit users |
| Period | March–May 2016 (31 days) |
| Files used | dailyActivity, sleepDay, dailyCalories, dailyIntensities |
| License | CC0 Public Domain |

**Habit Completion Thresholds:**

| Habit | Raw Column | Threshold | Completion Rate |
|---|---|---|---|
| Sleep | TotalMinutesAsleep | ≥ 420 min (7 hrs) | 24.5% |
| Workout | VeryActiveMinutes | ≥ 20 min | 32.9% |
| Steps | TotalSteps | ≥ 8,000 steps | 46.2% |
| Sedentary Limit | SedentaryMinutes | ≤ 600 min | 9.2% |

**Supplemented with** 50 synthetic users across 5 behavioral archetypes:
Consistent, Early Dropout, Weekend Hero, Inconsistent, Late Bloomer.

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/Swayon0507/habit-drop-predictor.git
cd habit-drop-predictor

# Create virtual environment
python -m venv venv

# Activate (Windows Git Bash)
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run Pipeline

```bash
# Step 1 — Download FitBit dataset
python - << 'EOF'
import kagglehub, shutil, os
path = kagglehub.dataset_download("arashnic/fitbit")
os.makedirs("data/raw/fitbit", exist_ok=True)
needed = ["dailyActivity_merged.csv","sleepDay_merged.csv",
          "dailyCalories_merged.csv","dailyIntensities_merged.csv","dailySteps_merged.csv"]
for root, dirs, files in os.walk(path):
    for f in files:
        if f in needed:
            shutil.copy(os.path.join(root,f), f"data/raw/fitbit/{f}")
            print(f"Copied: {f}")
EOF

# Step 2 — Run preprocessing
python src/features/preprocessing.py

# Step 3 — Train model with SMOTE
python src/models/train.py

# Step 4 — Run predictions
python src/models/predict.py
```

---

## 🖥️ Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

Opens at **http://localhost:8501**

---

## 🔬 Methodology

### Feature Engineering (11 features)

| Feature | Description | Why It Matters |
|---|---|---|
| `roll7_rate` | 7-day rolling completion rate | **#1 SHAP feature** — recent momentum |
| `roll14_rate` | 14-day rolling completion rate | Long-term baseline |
| `roll3_rate` | 3-day rolling completion rate | Short-term warning signal |
| `streak` | Consecutive days completed | Broken streak = high risk |
| `days_since_miss` | Days since last skip | Recent failure indicator |
| `weekend_penalty` | Weekday% minus Weekend% | **#2 SHAP feature** |
| `habit_age` | Days since tracking began | New and old habits at risk |
| `is_weekend` | Binary weekend flag | Day-type context |
| `dayofweek` | Day of week (0=Mon) | Temporal pattern |
| `mood` | Self-reported mood (1-10) | Psychological context |
| `habit_encoded` | Label-encoded habit name | Habit identity |

### Target Variable

A habit drop event (`will_drop = 1`) is automatically detected when:

```
R_past ≥ 0.60  AND  R_future ≤ 0.30
```

Where `R_past` = mean completion in prior 7 days, `R_future` = mean in next 7 days.
No manual labelling required — this is the core technical contribution.

### Model Architecture

- **XGBoost** classifier with `scale_pos_weight = 27.9`
- **SMOTE** oversampling to balance classes (7,437 → 7,437 each)
- **Isotonic regression calibration** via `CalibratedClassifierCV` (3-fold CV)
- **GroupShuffleSplit** — users never appear in both train and test sets

### Intervention Engine

| Condition | Suggested Action |
|---|---|
| Streak = 0 | Restart with a 2-day mini challenge |
| roll3_rate < 30% | Reduce difficulty immediately |
| weekend_penalty > 30% | Plan specifically for weekends |
| roll7_rate < 40% | Simplify habit for one week |
| Default | Add an extra reminder this week |

---

## 📊 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.13 | Core language |
| XGBoost | 3.2.0 | Gradient boosted classifier |
| scikit-learn | Latest | Calibration, metrics, splitting |
| imbalanced-learn | Latest | SMOTE oversampling |
| SHAP | Latest | Model explainability |
| Streamlit | 1.54.0 | Interactive dashboard |
| Plotly | Latest | Interactive charts |
| pandas / numpy | Latest | Data processing |
| joblib | Latest | Model serialization |

---

## 👨‍💻 Authors

**Swayon Bhunia** — MS Computer Science (AI Track)
**Prof. Dr. Sujoy Sikdar** — Supervisor

---

## 📄 References

1. Fogg, B.J. (2019). *Tiny Habits*. Houghton Mifflin Harcourt.
2. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
3. Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
4. Möbius. (2016). *FitBit Fitness Tracker Data*. Kaggle. CC0 Public Domain.
5. Lally, P. et al. (2010). How are habits formed. *European Journal of Social Psychology*.
