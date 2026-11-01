# ✈️ AeroMind: Jet Engine Remaining Useful Life (RUL) Prediction

Predict how many operating cycles a turbofan engine has left before failure, using classical machine learning on the NASA C-MAPSS FD001 dataset, and explore the results in an interactive Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey)

> ⚠️ **Research prototype.** Trained on simulated benchmark data. Not certified for real aviation maintenance decisions.

---

## 📌 Overview

Predictive maintenance replaces fixed-schedule servicing with decisions based on an engine's actual condition. This project:

- reads an engine's chronological sensor history,
- engineers **137 features** (differences and rolling statistics),
- predicts **RUL** with a tuned **Random Forest**,
- raises a maintenance alert when predicted RUL falls below **30 cycles**,
- serves everything through the **AeroMind** Streamlit app.

## 📊 Results (official NASA FD001 test set, 100 engines)

| Metric | Value |
|---|---|
| MAE | **18.09** cycles |
| RMSE | **25.95** cycles |
| R² | **0.61** |

**Maintenance alert (predicted RUL < 30):**

| True alerts | Predicted alerts | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| 25 | 20 | 19 | 1 | 6 | **0.95** | **0.76** |

<p align="center">
  <img src="assets/scatter.png" width="48%" alt="Actual vs predicted RUL">
  <img src="assets/errhist.png" width="48%" alt="Error distribution">
</p>

**Honest takeaways**

- The model tends to be **optimistic**: about 71% of errors are over-predictions, and 21 engines were over-predicted by more than 30 cycles. In maintenance, optimism is riskier than pessimism.
- Validation scores printed in the notebook (MAE ≈ 2.2, R² = 0) come from evaluating the *last cycle of training engines*, where RUL is always 0. That check is trivial; the **official test set is the real benchmark**.

## 🧠 Method

1. **Data:** NASA C-MAPSS FD001 (100 train / 100 test engines, 21 sensors, 3 operating settings). No missing values or duplicates.
2. **Cleaning:** removed 7 constant columns (`setting_3`, `sensor_1, 5, 10, 16, 18, 19`), leaving **17 inputs**.
3. **Target:** training RUL = last cycle of the engine − current cycle (uncapped). Test RUL comes from NASA's `RUL_FD001.txt`.
4. **Features (137):** 17 raw inputs, 17 one-cycle differences, rolling mean and std over windows of 5/10/20 cycles (102), plus `time_cycles`. Windows use present and past data only, so there is **no future leakage**.
5. **Split:** `GroupShuffleSplit` by engine (80/20); `GroupKFold` (5 folds) for tuning.
6. **Models compared:** Ridge, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting.
7. **Tuning:** `GridSearchCV`, best params: `max_depth=12`, `max_features=0.5`, `min_samples_leaf=5`.
8. **Final evaluation** on the untouched NASA test set.

### Most important features (permutation importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `sensor_14_rollstd_20` | 0.140 |
| 2 | `time_cycles` | 0.040 |
| 3 | `sensor_14_rollstd_10` | 0.034 |
| 4 | `sensor_4_rollmean_5` | 0.029 |
| 5 | `sensor_9_rollstd_20` | 0.022 |

Permutation importance is a diagnostic, not a causal explanation.

## 🗂️ Repository structure

```
.
├── app.py                              # Streamlit app (AeroMind)
├── rul_model.joblib                    # Trained Random Forest pipeline (~28 MB)
├── metadata.joblib                     # Feature lists, alert threshold, dataset info
├── requirements.txt
├── Jet_Engine_RUL_Prediction_Complete.ipynb   # Full training notebook
├── test_predictions.csv                # Per-engine actual vs predicted RUL
├── permutation_importance.csv          # Feature importance scores
├── test_FD001_selected_fixed.csv       # Sample engine histories for the app
└── assets/                             # Figures used in this README
```

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

> `scikit-learn==1.6.1` is pinned on purpose: the saved model must be loaded with the same version it was trained with.

## 🖥️ Using the app

The app has three pages:

- **Live RUL Prediction:** enter readings manually or upload a CSV for one engine. At least 2 consecutive cycles are required because the features depend on history.
- **Model Performance:** NASA test-set metrics and plots.
- **Feature Guide:** the inputs and engineered features.

Engine status: 🔴 **CRITICAL** (RUL < 30), 🟡 **WARNING** (RUL < 60), 🟢 **HEALTHY** otherwise.

**CSV format:** columns `time_cycles` plus the 17 inputs:
`setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21`.
An optional `unit_number` column lets you pick an engine. Try `test_FD001_selected_fixed.csv` as a sample.

## ⚠️ Limitations and future work

- Simulated data from a single subset (FD001: one operating condition, one fault mode).
- The model is sometimes over-optimistic; consider an asymmetric loss or a more conservative alert threshold.
- Ideas: cap RUL at 125 (common in the literature), sequence models (LSTM/GRU/Transformers), and the harder subsets FD002 to FD004.

## 📚 Dataset and references

- A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.
- A. Saxena, K. Goebel, D. Simon, N. Eklund (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008.

## 📄 License

Add a license of your choice (e.g. MIT) as a `LICENSE` file. The NASA dataset is subject to its own terms of use.
