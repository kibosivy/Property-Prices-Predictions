# Predicting House Prices

## Project Overview
This project focuses on predicting residential property sale prices using the Kaggle competition dataset **“House Prices: Advanced Regression Techniques.”**

It implements a complete end-to-end machine learning pipeline consisting of:
* Exploratory data analysis (EDA)
* Data cleaning and preprocessing
* Feature engineering
* Model development and evaluation
* Hyperparameter tuning
* Model explainability using XGBoost Feature Importance and SHAP
* Deep neural network (DNN) benchmarking
* Stacked ensemble modeling
* Final Kaggle submission
* **Deployment using Streamlit**

The goal is to build a highly accurate, generalizable, and interpretable model for structured real-estate data.
📌 **Competition Link:**
[https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## Dataset Description
| Dataset | Rows | Description |
| :--- | :--- | :--- |
| **Training Set** | 1,460 | Includes home attributes + target `SalePrice` |
| **Test Set** | 1,459 | Includes features only (`SalePrice` to be predicted) |

* **Total Features:** 79 structured predictors
* **Target Variable:** `SalePrice`

### Files Included
* `train.csv`
* `test.csv`
* `data_description.txt`
* `sample_submission.csv`

---

## Problem Statement
House price prediction plays an important role in real estate valuation, mortgage underwriting, and investment analysis.

The dataset contains a diverse set of numeric, ordinal, and categorical variables describing housing conditions, location, materials, and quality.
The challenge is to build a model capable of handling:
* High-dimensional structured data
* Strong non-linear interactions
* Mixed feature types
* Skewed target distributions
* Neighborhood-driven price variability

---

## Project Objectives
The main objectives include:
* Data cleaning and handling of missing values
* Domain-specific feature engineering
* Encoding categorical features
* Applying log transformation to stabilize `SalePrice`
* Training and evaluating multiple machine learning models
* Hyperparameter optimization for XGBoost
* Explainability using Feature Importance and SHAP
* Building a deep learning baseline
* Constructing a stacked ensemble
* Generating Kaggle-ready predictions
* **Deploying the model in a Streamlit application**

---

## Methodology

### 1. Data Understanding
Initial exploration involved descriptive statistics, outlier detection, distribution analysis, and correlation exploration.

**Key Findings**
* `SalePrice` was heavily skewed → required log-transformation
* Many missing values were structural (e.g., no basement)
* Location-related features strongly influenced pricing
* Quality and condition variables were ordinal

### 2. Data Cleaning and Preprocessing

#### Missing Value Handling
A domain-aware approach was used:

| Case | Method | Example Features |
| :--- | :--- | :--- |
| **Structural absence** | Filled with “None” or “NF” | `BsmtQual`, `GarageType` |
| **Numeric absence with structural meaning** | Filled with 0 | `GarageArea`, `TotalBsmtSF` |
| **Neighborhood-based** | Median per neighborhood | `LotFrontage` |
| **Misc categorical** | Mode imputation | `KitchenQual`, `MSZoning` |

#### Outlier Processing
* Extreme values capped using percentile thresholds
* Removed unrealistic combinations of size vs sale price

### 3. Feature Engineering
Constructed new domain-driven features to improve predictive power:

| Feature | Description |
| :--- | :--- |
| **TotalSF** | Total living area |
| **Total_Bathrooms** | Combined standardized bathroom count |
| **HouseAge** | House age at time of sale |
| **RemodAge** | Years since remodeling |
| **OverallQualArea** | Quality × size interaction |
| **QualityScore** | Combined structural condition |
| **TotalPorch** | Sum of all porch areas |

These engineered features improved correlations and boosted model performance.

### 4. Encoding Categorical Features
* Ordinal encoding applied to ordered quality variables
* One-hot encoding applied to nominal categories
* Alignment ensured consistent columns for train/test:

```python
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
````

### 5\. Target Transformation

To correct skewness:

```python
y = np.log1p(train["SalePrice"])
```

Inverse transform performed before submission:

```python
np.expm1(pred)
```

-----

## Model Development

### Models Implemented

  * Linear Regression (baseline)
  * Ridge & Lasso Regression
  * Random Forest
  * Gradient Boosting
  * **XGBoost** (primary model)
  * Deep Neural Network (DNN)
  * Stacked Ensemble (Ridge + Lasso + XGBoost)

#### Cross-Validation Results

| Model | CV R² |
| :--- | :--- |
| Random Forest | 0.8778 |
| Gradient Boosting | 0.8889 |
| **XGBoost** | **0.8946** |
| Stacked Ensemble | 0.8812 |
| DNN | 0.8342 |

### Hyperparameter Optimization (XGBoost)

#### Randomized Search

Best parameters included:

```python
{'subsample': 0.6, 'reg_lambda': 10, 'reg_alpha': 0.1,
 'n_estimators': 800, 'max_depth': 3,
 'learning_rate': 0.03, 'colsample_bytree': 1.0}
```

  * **CV R²:** 0.8965

#### Grid Search Refinement

```python
{'colsample_bytree': 0.7,
 'learning_rate': 0.04,
 'max_depth': 2,
 'n_estimators': 1000,
 'reg_alpha': 0.1,
 'reg_lambda': 9,
 'subsample': 0.7}
```

  * **Fine-Tuned CV R²:** **0.8998**

### Final XGBoost Results

  * **Validation R²:** **0.8785**
  * **Validation RMSE:** **0.1221**

**Interpretation**

  * Model explains \~88% of price variation
  * Balanced training/validation curves indicate strong generalization
  * Residuals centered around zero → unbiased predictions

### Model Explainability

#### Feature Importance (XGBoost)

Top impactful predictors:

  * `OverallQualArea`
  * `TotalSF`
  * `QualityScore`
  * `LotArea`
  * `GrLivArea`
  * `GarageYrBlt`

#### SHAP Analysis

SHAP revealed:

  * Higher quality, size, and updated condition increase price
  * Lower values reduce predicted price
  * Confirms domain-consistent model behavior

### Deep Neural Network (DNN)

**Architecture**

  * Dense layers: 256 → 128 → 64 → 1
  * ReLU activations
  * BatchNormalization + Dropout
  * Adam optimizer, MSE loss
  * Early stopping enabled

**Performance**

  * R²: 0.8342
  * RMSE: 0.1426
  * Deep learning underperformed compared to XGBoost, which is typical for structured tabular data.

### Stacked Ensemble

  * **Base Models:** Ridge, Lasso, XGBoost
  * **Meta Model:** Linear Regression
  * **R²:** 0.8812
  * **RMSE:** 0.1207
  * Provided stability but did not outperform XGBoost.

-----

## Final Predictions & Deployment

### Final Kaggle Submission

```python
y_test_pred = np.expm1(best_xgb.predict(X_test))
submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_test_pred})
submission.to_csv("submission.csv", index=False)
```

### Kaggle Leaderboard Performance

  * **Final Public Score:** 0.13744 RMSE
  * This validates strong performance and generalization of the tuned XGBoost model on unseen Kaggle test data.

### Streamlit Deployment

A full interactive prediction interface was deployed using Streamlit.

🔗 **Live App Link:**
[https://kibosivy-property-prices-predictions-app-mqyip5.streamlit.app/](https://kibosivy-property-prices-predictions-app-mqyip5.streamlit.app/)

**Features of the App**

  * Accepts both numeric and categorical inputs
  * Automatically handles one-hot encoding
  * Uses the trained XGBoost model
  * Predicts and displays the final `SalePrice`
  * Clean, structured UI

-----

## Technologies & Tools

  * **Python** 3.13
  * `pandas`, `NumPy`
  * `scikit-learn`
  * **XGBoost**
  * `TensorFlow` / `Keras`
  * **SHAP**
  * `Matplotlib`, `Seaborn`
  * **Streamlit**
  * `Git` & `GitHub`

-----

## Conclusion

This project presents a full, production-ready machine learning pipeline for residential house price prediction. Using strong preprocessing, carefully designed engineered features, and extensive model tuning, the final **XGBoost** model delivered high accuracy, strong generalization, and interpretable insights.

The inclusion of a deep learning benchmark and a stacked ensemble enriched the analysis.

The **Streamlit deployment** demonstrates practical usability, enabling real-time predictions through a user-friendly application.

```
```
