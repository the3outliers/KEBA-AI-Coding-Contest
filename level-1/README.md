# KEBA AI Coding Contest 2026: Level 1 - EV Charging Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This repository contains the winning solution for **Level 1 of the KEBA AI Coding Contest 2026**. The objective is to forecast electric vehicle (EV) charging behavior to help stabilize power grids and optimize charging infrastructure. 

Given a time-series dataset of EV charging sessions, the task is a **Binary Classification** problem: predicting the *probability* that a specific user is actively charging their vehicle at a specific future timestep.

**Evaluation Metric:** [Brier Score](https://en.wikipedia.org/wiki/Brier_score) (Lower is better).

---

## Dataset & Exploratory Data Analysis (EDA)
The dataset consists of real, unfiltered KEBA charging hardware logs. 
* **Target Variable (`y_b`):** `1` if actively charging, `0` if idle.
* **Class Imbalance:** Extreme. The global charging rate is only **~5.77%**.
* **Key Insight (The Workplace Pulse):** Charging activity peaks heavily on weekdays between 07:00 and 11:00 AM, dropping to near-zero on weekends. This indicates a strong **workplace commuter** profile.
* **User Heterogeneity:** Individual users exhibit highly consistent, personalized charging habits.

---

## Feature Engineering
To translate raw timestamps (`ds`) into machine-readable behavioral clues, the following features were engineered:
1. `hour`: Hour of the day (0-23).
2. `day_of_week`: Day of the week (0-6).
3. `is_weekend`: Binary flag for Saturdays and Sundays.
4. `is_business_hours`: Binary flag for core working hours (08:00 - 17:00).
5. **`user_hour_prob` (The Magic Lever):** A target-encoded feature capturing the historical probability of a specific `unique_id` charging at a specific `hour`. 

---

## Validation Strategy: The Temporal Split
To strictly prevent **Future Leakage** (using future events to predict the past), standard random K-Fold cross-validation was discarded. 

Instead, a **Chronological Temporal Split (80/20)** was implemented. The model was trained exclusively on historical data and validated on a hidden "future" timeline, perfectly simulating the KEBA platform's evaluation environment.

---

## Modeling Approach
**Algorithm:** Extreme Gradient Boosting (`XGBClassifier`)
* **Objective:** `binary:logistic` (Outputting calibrated probabilities, not hard 0/1 classes, to minimize the Brier Score penalty).
* **Why not ARIMA?** While ARIMA was tested, it struggled with the binary nature of the target, required clipping outputs to `[0, 1]`, and lacked the scalability to efficiently handle multiple distinct user timelines simultaneously. XGBoost elegantly mapped the extracted temporal tabular features.

### Results
The XGBoost model successfully captured both macro-seasonal trends and micro-user habits, dramatically outperforming the naive baseline.

| Model | Evaluation Metric | Score |
| :--- | :--- | :--- |
| **Naive Baseline** (Predicting Global Mean) | Brier Score | `0.05622` |
| **XGBoost Classifier** | Brier Score | **`0.03660`** |

*Note: The model achieved roughly a **35% reduction in error** over the baseline.*

---

## 🚀 How to Run

1. **Install Dependencies:**

    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```

2. **Place Data:** Ensure `level_1_train.csv` and `level_1_test_submission.csv` are in the root directory.

3. **Execute Pipeline:** Run the Jupyter Notebook (or Python script) sequentially:
    * **Step 1:** Data Ingestion & Sanitization
    * **Step 2:** Feature Engineering (Temporal & Behavioral)
    * **Step 3:** Validation Splitting & Baseline Calculation
    * **Step 4:** XGBoost Model Training & Evaluation
    * **Step 5:** Inference & Submission Generation

4. **Output:** The script will generate `keba_level1_xgb_submission.csv` (semicolon delimited), ready for leaderboard upload.

5. Once uploaded we have to check the team position in leaderboard, if we are not at the top we might need to change the model and test it once again.
