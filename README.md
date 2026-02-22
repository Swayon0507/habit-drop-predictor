# 🎯 Habit Drop Predictor

> Proactive Habit Drop Detection and Intervention System Using Behavioral Time-Series Analysis

## Overview
A machine learning system that predicts which micro-habits a person is likely
to drop in the upcoming week and suggests personalized corrective interventions.

## Key Features
- Predicts habit drop risk with 88.88% AUC-ROC score
- Personalized intervention suggestions per habit
- Interactive Streamlit dashboard
- SHAP explainability showing why each habit is at risk
- Supports real + synthetic multi-user data

## Project Structure
```
habit-drop-predictor/
├── data/
│   ├── raw/                  ← original Kaggle dataset
│   └── processed/            ← engineered features + labels
├── notebooks/
│   └── 01_eda.ipynb          ← exploratory data analysis
├── src/
│   ├── features/
│   │   └── preprocessing.py  ← data pipeline
│   ├── models/
│   │   ├── train.py          ← XGBoost + calibration
│   │   └── predict.py        ← risk scoring + interventions
│   └── dashboard/
│       └── app.py            ← Streamlit dashboard
└── reports/figures/          ← saved visualizations
```

## Setup
```bash
git clone https://github.com/Swayon0507/habit-drop-predictor.git
cd habit-drop-predictor
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Run Pipeline
```bash
python src/features/preprocessing.py data/raw/90_day_habit_tracker.csv
python src/models/train.py
python src/models/predict.py
```

## Run Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Tech Stack
Python · XGBoost · SHAP · Streamlit · Plotly · scikit-learn · pandas

## Dataset
90-Day Habit Tracker for Personal Growth — Kaggle (synthetic dataset)

## Results
- AUC-ROC: 0.8888
- Brier Score: 0.0224
- 6 habits tracked: Sleep, Workout, Reading, Screen Limit, Budget, Journaling
