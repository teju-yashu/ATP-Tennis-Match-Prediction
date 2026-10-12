# ATP Tennis Match Prediction

An end-to-end machine learning project that predicts ATP tennis match outcomes using historical match data, player strength ratings, recent performance, surface-specific statistics, and serve-related features.

The project implements a **Random Forest binary classification model** with a chronological training and evaluation pipeline designed to reduce data leakage and simulate real-world pre-match predictions.

## Key Features

- **Dynamic Elo ratings:** Estimate player strength using historical match outcomes.
- **Surface-specific Elo:** Capture differences in player performance across hard, clay, and other court surfaces.
- **Recent form analysis:** Incorporate recent win rates over the last 5, 10, and 20 matches, along with recency-weighted form.
- **Head-to-head statistics:** Represent historical matchup performance, including surface-specific H2H information.
- **Ranking and player statistics:** Use ATP rankings, ranking points, player attributes, tournament performance, and serve statistics.
- **Leakage-aware feature engineering:** Generate pre-match features from historical information, with same-date player-state updates applied only after all features for that date have been generated.
- **Feature ablation:** Reduce the initial 27 engineered features to 22 based on experimental evaluation.
- **Threshold optimization:** Select a decision threshold of 0.54 using validation data.
- **Model evaluation:** Assess classification performance, probability quality, calibration, and prediction errors.
- **Error analysis:** Examine performance across court surfaces, ranking gaps, and prediction-confidence levels.

## Dataset and Evaluation

The project uses historical ATP match records covering 1997–2024.

| Dataset split | Period | Matches |
|---|---|---:|
| Training | 1997–2022 | 71,905 |
| Validation | 2023 | 2,766 |
| Testing | 2024 | 1,302 |

The chronological split ensures that the final evaluation uses a future season that was not used for model selection or threshold optimization.

## Final Model

- **Algorithm:** Random Forest Classifier
- **Input features:** 22
- **Number of trees:** 300
- **Maximum tree depth:** 10
- **Minimum samples per split:** 10
- **Minimum samples per leaf:** 5
- **Maximum features per split:** 0.5
- **Decision threshold:** 0.54

## Final Test Results

Evaluated on the unseen 2024 test set:

| Metric | Result |
|---|---:|
| Accuracy | 65.05% |
| Precision | 66.85% |
| Recall | 57.57% |
| F1-score | 61.86% |
| ROC-AUC | 0.7203 |
| Log loss | 0.6143 |
| Brier score | 0.2131 |

The results indicate moderate predictive performance, with scope for improvement in close matchups and surface-specific generalization.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- FastAPI for the prediction API
- React-based frontend

## Project Objective

The objective is to develop and evaluate a reproducible tennis match prediction pipeline that combines historical player statistics, dynamic strength ratings, careful temporal feature engineering, and systematic model evaluation.

The project's emphasis is on **methodology, leakage prevention, feature analysis, and honest evaluation on unseen data**, rather than claiming a new machine learning algorithm.

## Disclaimer

This project is intended for educational and analytical purposes. Predictions are probabilistic and are not guarantees of match outcomes.
