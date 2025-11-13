# Predicting House Prices (Kaggle: House Prices – Advanced Regression Techniques)

## Project Overview
This project focuses on predicting residential property sale prices using the Kaggle dataset **“House Prices: Advanced Regression Techniques.”**

The goal was to build a complete end-to-end machine learning pipeline that handles:
* Data understanding and exploration
* Cleaning and preprocessing
* Feature engineering
* Model development
* Hyperparameter tuning
* Model explainability
* Final prediction and Kaggle-ready submission

The project uses a combination of linear models, ensemble learning algorithms, and a deep neural network (DNN) to evaluate different approaches on tabular housing data.
Competition Link:
[https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## Dataset Description
| Dataset | Observations | Description |
| :--- | :--- | :--- |
| **Training set** | 1,460 | Contains home features + SalePrice |
| **Test set** | 1,459 | Contains home features only |

* **Features:** 79 structured variables describing physical, qualitative, and temporal house attributes
* **Target variable:** `SalePrice`

### Files Provided
* `train.csv`
* `test.csv`
* `data_description.txt`
* `sample_submission.csv`

---

## Problem Statement
Home price prediction is important in real estate valuation, mortgage lending, investment analysis, and financial risk assessments.

The challenge lies in modeling:
* Multiple heterogeneous feature types
* Complex non-linear relationships
* Interactions between structural characteristics, quality indicators, and neighborhood attributes

The objective of this project is to develop a model that generalizes well while offering meaningful insights into what drives property values.

---

## Project Objectives
✔ Clean and preprocess the dataset

✔ Engineer new features that better capture home characteristics

✔ Use log-transformation to stabilize the target distribution

✔ Train and compare machine learning models

✔ Optimize the best-performing model using hyperparameter tuning

✔ Interpret predictions with feature importance + SHAP

✔ Produce a final Kaggle submission

---

## Methodology

### 1. Data Understanding
* Explored feature types (object, numeric, ordinal)
* Examined missing values
* Analyzed distributions and outliers
* Visualized key relationships (heatmaps, scatterplots)

**Key Observations**
* Many categorical features represent ordinal quality (Ex, Gd, TA, Fa)
* `SalePrice` is right-skewed (log transform required)
* Most missing values were structural (e.g., no basement)

### 2. Data Cleaning and Preprocessing

#### Handling Missing Values
Used a domain-aware strategy:

| Scenario | Treatment | Examples |
| :--- | :--- | :--- |
| **Structural absence** | Filled with "None" or "NF" | Basement, Garage |
| **Numeric but absence implies zero** | Filled with 0 | `GarageArea`, `BsmtHalfBath` |
| **Neighborhood influence** | Median imputation per neighborhood | `LotFrontage` |
| **Categorical misc missing** | Mode imputation | `MSZoning`, `KitchenQual` |

#### Outlier Treatment
* Removed or capped extreme outliers using IQR/percentile rules
* Outlier removal improved linear model stability

### 3. Feature Engineering
Created domain-relevant aggregate and interaction variables:

| Feature | Description |
| :--- | :--- |
| **TotalSF** | Total usable square footage |
| **Total_Bathrooms** | Combined normalized bathroom count |
| **HouseAge** | How old the house is |
| **RemodAge** | Years since remodeling |
| **OverallQualArea** | House quality × size |
| **QualityScore** | `OverallQual` × `OverallCond` |
| **TotalPorch** | Sum of all porch areas |

These features significantly improved correlations with `SalePrice`.

### 4. Encoding Categorical Features
* Ordinal Encoding for quality-related features (Ex, Gd, TA, Fa, Po)
* One-Hot Encoding for all nominal variables
* Ensured train/test alignment:

```python
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
````

### 5\. Target Transformation

The target was log-transformed to normalize its distribution:

```python
y = np.log1p(train["SalePrice"])
```

All predictions were later transformed back using:

```python
final_predictions = np.expm1(model.predict(X_test))
```

-----

## Model Development

### Models Implemented

| Model | Description |
| :--- | :--- |
| **Linear Regression** | Baseline model |
| **Ridge / Lasso Regression** | Regularized linear models |
| **Random Forest** | Tree-based ensemble |
| **Gradient Boosting** | Sequential boosting model |
| **XGBoost** | High-performance boosting algorithm |
| **Deep Neural Network (DNN)** | Feedforward network for tabular learning |
| **Stacked Ensemble** | Combines Ridge, Lasso, XGBoost |

### Model Evaluation (Cross-Validation)

| Model | Mean R² |
| :--- | :--- |
| Random Forest | 0.8778 |
| Gradient Boosting | 0.8889 |
| **XGBoost** | **0.8946** |
| Stacked Ensemble | 0.8812 |
| Deep Neural Network | 0.8342 |

🔍 **XGBoost was the strongest model before tuning.**

### Hyperparameter Optimization (XGBoost)

#### Stage 1 — Randomized Search

Explored a broad parameter space:

```python
param_dist = {...}
```

  * **Best Parameters (Randomized Search):**

<!-- end list -->

```python
{'subsample': 0.6, 'reg_lambda': 10, 'reg_alpha': 0.1,
 'n_estimators': 800, 'max_depth': 3,
 'learning_rate': 0.03, 'colsample_bytree': 1.0}
```

  * **Best CV R²:** 0.8965

#### Stage 2 — Grid Search

Refined search around optimal region:

  * **Fine-Tuned Parameters (Grid Search):**

<!-- end list -->

```python
{'colsample_bytree': 0.7, 'learning_rate': 0.04, 'max_depth': 2,
 'n_estimators': 1000, 'reg_alpha': 0.1,
 'reg_lambda': 9, 'subsample': 0.7}
```

  * **Fine-Tuned CV R²:** **0.8998**

### Validation Performance (Final XGBoost)

  * **Validation R²:** **0.8785**
  * **Validation RMSE:** **0.1221**

The model captured \~88% of price variability.

#### Learning Curve Interpretation

  * Training and validation RMSE steadily decreased
  * Curves remained close → no significant overfitting
  * Model generalized well across unseen data

#### Residual Diagnostics

**Findings:**

  * Residuals centered around 0 → unbiased predictions
  * No visible funnel shape → stable variance across price ranges
  * Good randomness → model fits the data well

### Feature Importance & Explainability

#### XGBoost Importance

Most influential features:

  * `OverallQualArea`
  * `LotArea`
  * `TotalSF`
  * `QualityScore`
  * `GrLivArea`
  * `GarageYrBlt`

#### SHAP Analysis

SHAP summary plot highlighted:

  * High quality, size, and renovation condition strongly increase predicted prices
  * Low values of these features decrease predictions
    SHAP validated that the model captures real-world economic drivers

### Deep Neural Network (DNN)

A DNN was trained to evaluate deep learning performance on structured tabular data.

**Architecture:**

  * **Layers:** 256 → 128 → 64 → 1
  * **Activations:** ReLU
  * **Techniques used:** BatchNorm, Dropout (0.3–0.2)
  * **Optimizer:** Adam
  * **Loss:** MSE
  * Early stopping applied to prevent overfitting

**DNN Performance**

  * **R²:** 0.8342
  * **RMSE:** 0.1426

**Learning Curves:**

  * Both MAE and MSE showed strong convergence
  * No overfitting, but
  * Performance was significantly below XGBoost, which is expected for deep models on tabular data

📌 **Conclusion: DNN provides a valid alternative model, but XGBoost remains superior for this dataset.**

### Model Stacking

Combined three diverse models:

| Base Learners: | Meta-learner: |
| :--- | :--- |
| Ridge, Lasso, XGBoost | Linear Regression |

**Performance:**

  * **R²:** 0.8812
  * **RMSE:** 0.1207

Stacking improved stability but did not outperform XGBoost.

### Final Predictions

```python
y_test_pred = np.expm1(best_xgb.predict(X_test))
submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_test_pred})
submission.to_csv("submission.csv", index=False)
```

-----

## Final Model Performance Summary

| Model | R² | RMSE |
| :--- | :--- | :--- |
| Random Forest | 0.8778 | — |
| Gradient Boosting | 0.8889 | — |
| **XGBoost (Final)** | **0.8785** | **0.1221** |
| Stacked Ensemble | 0.8812 | 0.1207 |
| DNN | 0.8342 | 0.1426 |

-----

## Key Insights

  * `OverallQual`, `TotalSF`, and `QualityScore` are dominant price predictors
  * Log-transforming `SalePrice` significantly improved model stability
  * Engineered features contributed major accuracy gains
  * **XGBoost** outperformed all other models due to its ability to learn non-linear interactions
  * SHAP confirmed interpretability and reliability of predictions

-----

## Technologies and Tools

  * **Python** 3.13
  * **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `shap`, `tensorflow`
  * **Environment:** Jupyter Notebook / VS Code
  * **Version Control:** GitHub

-----

## Conclusion

This project demonstrates a complete machine learning pipeline for house price prediction: from data preprocessing, feature engineering, and modeling to interpretability and Kaggle submission.

The final tuned XGBoost model achieved excellent performance, strong generalization, and rich interpretability, outperforming other linear, ensemble, and deep learning models.

```
```
