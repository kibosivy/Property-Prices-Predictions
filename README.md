# Predicting House Prices (Kaggle: House Prices – Advanced Regression Techniques)

## Project Overview
This project focuses on predicting residential property sale prices using the Kaggle dataset **“House Prices: Advanced Regression Techniques.”**

The objective is to build a **robust regression pipeline** capable of accurately estimating home values through systematic data cleaning, feature engineering, and the application of advanced machine learning algorithms.
The project follows a complete data science workflow, covering data understanding, preprocessing, feature engineering, model development, hyperparameter optimization, validation, and interpretability.
Competition Link: [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## Dataset Description
| Dataset | Observations | Description |
| :--- | :--- | :--- |
| **Training set** | 1,460 | Includes property features and sale prices |
| **Test set** | 1,459 | Includes property features (SalePrice to be predicted) |

Source: Kaggle
* **Features:** 79 explanatory variables describing structural, qualitative, and temporal aspects of homes in Ames, Iowa.
* **Target variable:** `SalePrice` (continuous numeric variable).

### Files Included
* `train.csv`
* `test.csv`
* `data_description.txt`
* `sample_submission.csv`

---

## Problem Statement
Accurately estimating the sale price of a property is an important problem in real estate analytics, investment risk assessment, and mortgage valuation.

Given the wide range of property attributes such as lot size, construction quality, and location, the challenge is to design a model that can generalize effectively and explain the underlying factors driving price variability.

---

## Project Objectives
* Clean and preprocess the dataset by handling missing values and encoding categorical features appropriately.
* Engineer domain-specific features that improve model accuracy.
* Apply log transformation to stabilize the target variable distribution.
* Build, tune, and compare several machine learning models for predictive performance.
* Evaluate models using multiple regression metrics.
* Interpret the final model using feature importance and SHAP analysis.
* Generate a final submission file for Kaggle evaluation.

---

## Methodology

### 1. Data Understanding
* Examined the dataset structure using `info()` and `describe()`.
* Identified missing values, outliers, and feature data types.
* Visualized distributions and relationships using histograms, scatterplots, and boxplots.
* Assessed correlations between numeric features using a heatmap.

**Key Findings**
* Several categorical variables used ordinal quality ratings (e.g., Ex, Gd, TA, Fa).
* `SalePrice` exhibited right-skewed distribution, requiring transformation.
* Missing values were mainly structural (e.g., no basement, no garage).

### 2. Data Cleaning and Preprocessing

#### Handling Missing Values
A domain-specific approach was used based on the meaning of missingness:

| Type | Approach | Example |
| :--- | :--- | :--- |
| **Structural Missingness** | Filled with "None" or "NF" to denote "Not Found". | `GarageType`, `BsmtQual`, `PoolQC` |
| **Numeric Missingness** | Replaced with 0 when absence implied none. | `MasVnrArea`, `GarageArea`, `TotalBsmtSF` |
| **Neighborhood-Based Imputation** | Filled using the median value per neighborhood. | `LotFrontage` |
| **Mode Imputation** | Replaced isolated missing categorical values with the mode. | `MSZoning`, `KitchenQual` |

This ensured that structural absence (e.g., no basement) was not treated as random missing data.

#### Outlier Treatment
* Outliers were detected using boxplots and scatterplots.
* Extreme or unrealistic observations (e.g., exceptionally large `GrLivArea` with low `SalePrice`) were capped using the IQR or 99th percentile method.

### 3. Feature Engineering
New variables were created to capture interactions and aggregated property characteristics:

| Feature | Formula | Description |
| :--- | :--- | :--- |
| **TotalSF** | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Total living area |
| **Total_Bathrooms** | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | Standardized total bathrooms |
| **RemodAge** | `YrSold - YearRemodAdd` | Time since renovation |
| **HouseAge** | `YrSold - YearBuilt` | Building age |
| **OverallQualArea** | `OverallQual * TotalSF` | Quality × Size interaction |
| **QualityScore** | `OverallQual * OverallCond` | Combined structural condition |
| **TotalPorch** | `OpenPorchSF + EnclosedPorch + ScreenPorch + 3SsnPorch` | Total porch area |

These engineered features improved the model’s ability to capture property size, age, and condition effects.

### 4. Encoding Categorical Variables
* Ordinal encoding was applied to ordered quality-related variables using numeric scales.
* One-hot encoding was applied to nominal variables such as `Neighborhood` and `RoofStyle`.
* Train and test datasets were aligned to ensure identical columns:

```python
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
````

### 5\. Target Transformation

The target variable `SalePrice` was log-transformed to normalize its distribution and stabilize variance:

```python
y = np.log1p(train["SalePrice"])
```

Before saving the final predictions, the inverse transform was applied:

```python
y_test_pred = np.expm1(predictions)
```

-----

## Model Development

### Models Implemented

| Model | Type | Description |
| :--- | :--- | :--- |
| **Linear Regression** | Baseline | Establishes a reference performance |
| **Ridge & Lasso Regression** | Regularized Linear | Mitigates multicollinearity and overfitting |
| **Random Forest** | Ensemble (Bagging) | Reduces variance using multiple trees |
| **Gradient Boosting** | Ensemble (Boosting) | Sequentially improves weak learners |
| **XGBoost** | Optimized Boosting | Captures complex non-linear relationships efficiently |
| **Stacked Model** | Meta-Ensemble | Combines base models for improved generalization |

### Model Evaluation

5-fold cross-validation was used to evaluate generalization performance.

| Model | Mean R² |
| :--- | :--- |
| Random Forest | 0.8778 |
| Gradient Boosting | 0.8889 |
| **XGBoost** | **0.8946** |
| Stacked Ensemble | 0.8812 |

**Interpretation:**
XGBoost demonstrated superior performance due to its ability to model complex non-linear relationships and regularization mechanisms that prevent overfitting.

### Hyperparameter Optimization (XGBoost)

#### Stage 1: Randomized Search

A broad search was conducted to explore key parameters:

```python
param_dist = {
  'n_estimators':[400,800,1200],
  'learning_rate':[0.01,0.03,0.05],
  'max_depth':[3,4,5],
  'subsample':[0.6,0.8,1.0],
  'colsample_bytree':[0.6,0.8,1.0],
  'reg_lambda':[1,5,10],
  'reg_alpha':[0,0.01,0.1]
}
```

  * **Best Parameters:** `{'subsample': 0.6, 'reg_lambda': 10, 'reg_alpha': 0.1, 'n_estimators': 800, 'max_depth': 3, 'learning_rate': 0.03, 'colsample_bytree': 1.0}`
  * **Best CV R²:** 0.8965

#### Stage 2: Grid Search

Refined the search around optimal regions.

  * **Final Parameters:** `{'colsample_bytree': 0.7, 'learning_rate': 0.04, 'max_depth': 2, 'n_estimators': 1000, 'reg_alpha': 0.1, 'reg_lambda': 9, 'subsample': 0.7}`
  * **Fine-Tuned CV R²:** **0.8998**

### Validation Results

  * **Validation R²:** **0.8785**
  * **Validation RMSE:** **0.1221**

**Interpretation:**
The model explains approximately 88% of the variation in sale prices with a relatively low prediction error.

The learning curve demonstrated good convergence between training and validation errors, indicating balanced bias and variance.

#### Residual Analysis

  * The residual distribution was centered around zero, indicating unbiased predictions.
  * The residuals vs predicted values plot showed random scatter, suggesting homoscedasticity and stable performance across the prediction range.

### Feature Importance and Explainability

#### XGBoost Feature Importance

Key predictive variables included:

  * `OverallQualArea`
  * `TotalSF`
  * `QualityScore`
  * `GarageYrBlt`
  * `GrLivArea`
  * `LotArea`

These features represent the primary structural and qualitative attributes influencing sale price.

#### SHAP Analysis

The SHAP summary plot highlighted:

  * Higher values of `OverallQual`, `TotalSF`, and `QualityScore` positively impact predicted prices.
  * Older or smaller houses with lower quality attributes contribute negatively.

SHAP analysis validated that the model captures meaningful and domain-relevant relationships.

### Model Stacking

A stacked ensemble was developed to enhance model robustness.

| Base Models | Meta-Model |
| :--- | :--- |
| Ridge, Lasso, XGBoost | Linear Regression |

The stacked model leveraged diverse learning patterns. Although XGBoost remained the best individual performer, stacking provided a small improvement in overall stability.

### Final Predictions

Predictions were generated on the cleaned and preprocessed test data:

```python
y_test_pred = np.expm1(best_xgb.predict(X_test))
submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_test_pred})
submission.to_csv("submission.csv", index=False)
```

The resulting file, `submission.csv`, contained predicted sale prices corresponding to each property ID in the test set.

-----

## Model Performance Summary

| Model | R² | RMSE |
| :--- | :--- | :--- |
| Random Forest | 0.8778 | — |
| Gradient Boosting | 0.8889 | — |
| **XGBoost** | **0.8946** | **0.1221** |
| Stacked Ensemble | 0.8812 | 0.1207 |

**Conclusion:**
XGBoost achieved the best predictive accuracy and stability, balancing bias and variance effectively.

The model generalizes well across unseen data and aligns with domain expectations.

-----

## Technologies and Tools

  * **Programming Language:** Python 3.13
  * **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`
  * **Environment:** Jupyter Notebook / VS Code
  * **Version Control:** Git and GitHub

-----

## Key Insights

  * Property quality, total living area, and renovation recency are the strongest determinants of sale price.
  * Applying a log transformation to `SalePrice` improved model performance and stability.
  * Feature engineering (particularly `OverallQualArea` and `QualityScore`) significantly enhanced predictive power.
  * XGBoost provided superior results due to its ability to handle complex, non-linear interactions.
  * SHAP analysis confirmed the interpretability and trustworthiness of model predictions.

-----

## Conclusion

This project demonstrates a comprehensive machine learning workflow for house price prediction.

It covers data preprocessing, advanced feature engineering, model selection, hyperparameter tuning, and explainability.
The final XGBoost model achieved strong performance with an $\text{R}^2$ of 0.8785 on the validation set and provided interpretable insights into the key drivers of property prices.

The project highlights the importance of domain-driven feature creation and the use of ensemble methods for high-dimensional, tabular datasets.

```
```
