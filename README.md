# House Price Prediction

**Advanced Regression Techniques | Ridge Regression Model | SHAP Explainability | Streamlit Deployment**

---

## Project Overview

This project tackles the classic Kaggle competition **“House Prices: Advanced Regression Techniques”**, using a complete end-to-end machine learning workflow to build a clean, interpretable, and high-performing model for predicting housing sale prices.

The complete workflow includes:
* Exploratory Data Analysis (EDA)
* Missing value treatment and outlier handling
* Domain-driven **feature engineering**
* Ordinal + one-hot encoding
* Model development and cross-validation
* Hyperparameter tuning (**GridSearchCV** + validation curves)
* Learning curve + error analysis
* **SHAP explainability**
* **Streamlit deployment**
* Final Kaggle submission (Public Score: **0.15142** RMSLE)

### Competition Link:
[https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## Dataset Description

The project utilizes the Ames Housing Dataset.

| File | Rows | Description |
| :--- | :--- | :--- |
| `train.csv` | 1,460 | Features + Target (`SalePrice`) |
| `test.csv` | 1,459 | Features only (prediction required) |
| `data_description.txt` | — | Feature dictionary |
| `sample_submission.csv` | — | Submission format |

* **Total features:** 79 structured predictors
* **Target:** `SalePrice` (highly skewed, requires log-transformation)

---

## Problem Statement

Accurately predicting housing prices is crucial for **real estate valuation**, **loan underwriting**, and **investment decisions**.

The main challenges addressed were:
* Handling many mixed feature types (numeric, nominal, ordinal).
* Dealing with skewed numeric distributions and strong outliers.
* Mitigating **strong correlation + multicollinearity** among features.
* Resolving structural and non-structural missing values.
* Modeling non-linear and neighborhood-driven price variation.

---

## Data Cleaning & Preprocessing

### ✔ Missing Value Handling (Domain-Aware)

Missing values were handled based on the feature's domain context:

* **Structural Absences (Categorical):** Filled with **"None"** (e.g., `BsmtQual`, `GarageType`, `PoolQC`, `Fence`) to signify the absence of that feature.
* **Numeric Structural Absences:** Filled with **0** (e.g., `TotalBsmtSF`, `GarageArea`, `GarageCars`).
* **Neighborhood-based Imputation:** `LotFrontage` filled with the **median** per neighborhood.
* **Mode/Median Imputation:** Small-missing categorical features filled with the mode; small-missing numeric features filled with the median.

### Outlier Handling

Outliers were carefully managed to ensure model stability:
* **Removed** unrealistic outliers (e.g., very large homes sold unusually cheap).
* Used **IQR-based trimming** on highly skewed numeric features.
* **Winsorized** extreme percentiles for stability.

### Feature Engineering

Meaningful, domain-driven features were created to enhance model performance and interpretability:

| Feature | Description |
| :--- | :--- |
| `TotalSF` | Total square footage (basement + floors) |
| `Total_Bathrooms` | Combined full/half baths (above + basement) |
| `HouseAge` | Years since built |
| `RemodAge` | Years since remodel |
| `OverallQualArea` | Quality score (`OverallQual` $\times$ `TotalSF`) |
| `QualityScore` | Overall quality $\times$ condition |
| `TotalPorch` | Combined porch areas |

### Encoding Strategy

* **Ordinal Encoding:** Applied to ordered quality variables (e.g., `ExterQual`, `KitchenQual`, `BsmtQual`, `FireplaceQu`).
* **One-Hot Encoding:** Applied to nominal categorical variables (e.g., `Neighborhood`, `MSZoning`, `HouseStyle`, `Exterior1st`). Train/Test alignment was automatically handled to ensure consistent feature sets.

---

## Model Development

### Explored Models:
* Linear Regression
* Lasso
* **Ridge Regression** ✔ (Selected Model)
* ElasticNet
* Random Forest
* XGBoost
* Support Vector Regression

### Why Ridge Regression?

**Ridge Regression** was selected as the final model due to its stable, consistent, and generalizable performance. It excels because it:
* Handles the high level of **multicollinearity** well.
* Performs smoothly with **80+ encoded features**.
* Remains highly **interpretable** (unlike complex tree models).

### Hyperparameter Tuning (GridSearchCV)

The model was tuned using **GridSearchCV** over the following parameters:
* Polynomial degree: `[1, 2]`
* Alpha values ($\alpha$, regularization strength): `logspace(0, 6)`

#### Best Model Parameters:
* **Degree:** 1
* **Optimal Alpha ($\alpha$):** $\approx 2.07$
* **Mean CV RMSE:** $\approx 18,023$
* **Mean CV R²:** $\approx 0.9127$

---

## 🧪 Model Diagnostics

### ✔ Validation Curves
The validation curves demonstrated optimal performance around the selected alpha, illustrating the trade-off between bias and variance. 

### ✔ Learning Curve
The learning curve showed **good generalization** with high training and cross-validation scores converging quickly, indicating **no significant high variance or high bias** issues.

### ✔ Error Analysis
* **Residuals** were centered around zero (unbiased predictions).
* **Standardized residuals** showed no signs of **heteroscedasticity**.
* The **Actual vs. Predicted** plot showed a tight clustering around the $45^\circ$ line.

---

## Model Explainability (SHAP)

**SHAP (SHapley Additive exPlanations)** was used to interpret the model's predictions and understand feature contributions. 

### Key Drivers of Price:
1.  **`OverallQualArea`** (Strongest driver, quality-weighted area)
2.  **`GrLivArea`** (Above ground living area)
3.  **`TotalSF`** (Total square footage)
4.  **`TotalBsmtSF`** (Total basement square footage)
5.  **`LotArea`**
6.  **`GarageCars`**

The contributions were consistent with real-world real estate behavior: higher values for these features led to an increase in predicted sale price.

---

## Final Submission

The tuned Ridge model was retrained on the full training dataset and used to generate the final predictions.

* **Kaggle Score:** **0.15142** (RMSE)
* Result: Strong performance and excellent generalization on unseen test data.

---

## Streamlit Deployment

A lightweight, interactive application was built using **Streamlit** for real-world usability.
* Users can input key home characteristics via a clean UI.
* The model predicts the sale price instantly.
* Feature contributions (via SHAP) can optionally be displayed to provide transparency.
* https://kibosivy-property-prices-predictions-app-mqyip5.streamlit.app/

---

## Technologies Used

* **Python** 3.13
* `pandas`, `NumPy`
* `scikit-learn`
* `SHAP`
* `matplotlib`, `seaborn`
* `Streamlit`
* `Git & GitHub`

---

## Conclusion

This project successfully demonstrates a complete and production-ready machine learning pipeline for housing price prediction. Through strong preprocessing, robust feature engineering, Ridge regression tuning, thorough diagnostics, and SHAP explainability, the final model achieved reliable and interpretable performance, which was then deployed via Streamlit.
