# Predicting House Prices (Kaggle: House Prices – Advanced Regression Techniques)

## Project Overview
This project tackles the task of predicting residential property sale prices using the Kaggle competition dataset **“House Prices: Advanced Regression Techniques.”**  
The goal is to build a robust regression pipeline using advanced feature engineering, dimensionality reduction, and machine learning algorithms to generate accurate predictions and gain insights into the factors influencing price variation.

 **Competition Link:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

---

## 🧾 Dataset
**Source:** Kaggle  
**Training set:** 1,460 observations  
**Test set:** 1,459 observations  
**Features:** 79 explanatory variables describing physical, temporal, and qualitative aspects of homes in Ames, Iowa.  
**Target variable:** `SalePrice`

**Files included:**
- `train.csv`
- `test.csv`
- `data_description.txt`
- `sample_submission.csv`

---

## Problem Statement
Estimating the sale price of a house is a typical regression problem with real-world applications in real estate analytics, mortgage risk assessment, and investment decision-making.  
Given the rich set of property attributes, the challenge is to develop a predictive model that generalises well and uncovers the underlying drivers of price variability.

---

## Project Objectives
- Clean and preprocess the dataset, handling missing values and encoding categorical features.  
- Apply **Principal Component Analysis (PCA)** for dimensionality reduction and to handle multicollinearity.  
- Engineer meaningful new features that capture home characteristics.  
- Build and compare machine learning models to predict `SalePrice`.  
- Evaluate model performance using regression metrics and interpret key insights.  
- Generate a final **Kaggle submission** file.

---

## Methodology

### ** Data Understanding**
- Explore data structure, feature types, and value distributions.  
- Identify missing values and outliers.  
- Summarise key statistics for both numeric and categorical variables.

### ** Data Cleaning & Preprocessing**
- **Missing values:**  
  - Fill “None” for categorical features where missing implies absence.  
  - Fill 0 for numeric features such as basement or garage area when missing implies none.  
  - Median imputation for numeric columns like `LotFrontage` based on neighborhood.  
- **Outlier handling:** Remove or cap extreme price or area outliers.  
- **Encoding:**  
  - Ordinal encoding for quality/condition features.  
  - One-hot encoding for nominal features.  
- **Feature alignment:** Ensure train and test sets have identical columns.

### ** Feature Engineering**
- Create new aggregated or derived features such as total living space, total bathrooms, house age, renovation age, and combined quality-area metrics.  
- Apply log transformation to skewed features (including `SalePrice`).

### ** Dimensionality Reduction (PCA)**
- Scale all numeric features using `StandardScaler`.  
- Apply **PCA (n_components=0.99)** to retain 99% of total variance.  
- Visualise explained variance ratio to understand how much information each component holds.  
- The result produced 173 principal components, ensuring dimensionality reduction without significant information loss.

### ** Model Building**
Models implemented and compared include:  
- **Linear Regression** – baseline model  
- **Ridge & Lasso Regression** – regularised linear models  
- **Random Forest Regressor** – tree-based ensemble  
- **XGBoost Regressor** – gradient boosting model  
- **Neural Network (Keras/TensorFlow)** – deep learning approach  

Each model was evaluated using **RMSE**, **MAE**, and **R²** scores to determine predictive accuracy and model generalisation.

---

### ** Model Evaluation & Interpretation**
Evaluation metrics used:  
- **RMSE (Root Mean Squared Error)** – main Kaggle competition metric  
- **MAE (Mean Absolute Error)** – measures average prediction error  
- **R² Score** – measures the proportion of variance explained by the model  

Interpretation tools included:  
- Predicted vs Actual plots  
- Residual distribution analysis  
- Feature importance interpretation (for tree-based models)  

---

### ** Submission & Insights**
The final submission file contained the model’s predicted `SalePrice` values for each `Id` in the test dataset.

**Key insights identified:**
- Property quality (`OverallQual`) and total living area (`GrLivArea`) were the strongest predictors of sale price.  
- Houses that were newly built or recently renovated showed higher valuations.  
- Neighborhood, exterior material, and basement size also had notable influence.  
- Feature reduction through PCA improved model efficiency and stability without major accuracy loss.

---

## Technologies & Tools
- **Languages:** Python  
- **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, TensorFlow/Keras  
- **Environment:** Jupyter Notebook / VS Code  
- **Version Control:** Git & GitHub  

---

## Learning Goals
- Gain hands-on experience with **real-world regression datasets**.  
- Understand how **feature engineering** and **dimensionality reduction** improve model performance.  
- Compare **traditional machine learning** and **deep learning** models for predictive tasks.  
- Strengthen **data storytelling** and **model interpretation** skills for real-world applications.  



