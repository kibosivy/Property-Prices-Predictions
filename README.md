# House Price Prediction - Advanced Regression Techniques

---

## Project Overview

Predicting house prices accurately is a critical task in real estate analytics, helping sellers set reasonable expectations, buyers make informed decisions, and developers assess market value. Housing prices are influenced by a combination of structural characteristics, quality indicators, amenities, and location-driven variables. Because these factors interact in complex ways, a well-designed machine learning model can provide more reliable price estimates than simple heuristics.


---

## Problem Statement

The dataset contains hundreds of residential properties with over 80 diverse features—including numerical measurements, categorical descriptors, quality ratings, and engineered features.

The main challenges addressed were:
* Handling many mixed feature types (numeric, nominal, ordinal).
* Dealing with skewed numeric distributions and strong outliers.
* Mitigating **strong correlation + multicollinearity** among features.
* Resolving structural and non-structural missing values.
* Modeling non-linear and neighborhood-driven price variation.

---

**The complete workflow includes:**
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

## Data Cleaning & Preprocessing

### Missing Value Handling

The House Prices dataset contains a mixture of structural, informational, and random missing values.
Because these missing values arise from different underlying causes, applying a single imputation strategy would distort the data.

<img width="2190" height="989" alt="image" src="https://github.com/user-attachments/assets/0bddd549-84b4-4032-9ac8-6e1c42af27b8" />

The heatmap visualizes the pattern of missing values across all features in the dataset.
A few features — such as PoolQC, Alley, Fence, FireplaceQu, and Garage-related attributes — show concentrated missingness, which aligns with structural absence (many homes simply do not have pools, alleys, or garages).

To ensure correct interpretation and preserve real-estate logic, missing values were handled using four different strategies, each based on the meaning and behavior of the feature.

🔹 1. Structural Missingness (Feature Does Not Exist)

Many NaNs occur because the home does not have a basement, garage, pool, alley access, masonry veneer, or fireplace.

To avoid incorrect imputations, we systematically verified structural missingness:

✔ Garage-related NaNs

If GarageType was NaN:
→ GarageArea, GarageCars, GarageYrBlt, GarageQual, and GarageCond were also NaN or zero.
This confirms the home has no garage.

Imputation:

Categorical: "None"

Numeric: 0



---

✔ Basement-related NaNs

If BsmtQual, BsmtCond, or BsmtExposure was NaN:
→ BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath were zero or NaN.
This confirms the home has no basement.

Imputation:

Categorical: "None"

Numeric: 0



---

✔ Fireplace NaNs

If FireplaceQu was NaN:
→ Fireplaces = 0
This means the home has no fireplace.

Imputation:

Categorical: "None"



---

✔ Pool NaNs

If PoolQC was NaN:
→ PoolArea = 0
Meaning the home has no pool.

Imputation:

Categorical: "None"

Numeric: 0



---

✔ Masonry veneer NaNs

If MasVnrType was NaN:
→ MasVnrArea was 0 or NaN
This indicates no brick/stone veneer.

Imputation:

MasVnrType = "None"

MasVnrArea = 0 (or median for rare cases)



---

✔ Fence, Alley, MiscFeature

These features are mostly NaN because the majority of homes do not have:

fencing

alley access

miscellaneous external structures


There are no related numeric features, so absence = no feature.

Imputation: "None"


---

🔹 2. Numerical Features That Equal Zero When Absent

For structural numeric features:

Feature	Meaning of NaN

GarageArea, GarageCars	No garage
TotalBsmtSF, BsmtFinSF1/2	No basement
MasVnrArea	No masonry


Imputation: 0
This is correct because these quantities physically equal zero when absent.


---

🔹 3. LotFrontage

LotFrontage is the only numerical feature where missingness is not structural.
It reflects lot width facing the street and varies by neighborhood.

To impute realistically:

We computed median LotFrontage per Neighborhood

If missing, fill with that neighborhood’s median

If still missing: fill with overall median


---

🔹 4. Small-Missing Categorical Features → Mode

For features with very low missingness (≤ 1%), such as:

Electrical

MSZoning

Utilities

Exterior1st, Exterior2nd

SaleType

Functional


We filled them with the mode (most common value).
We confirmed that the features do exist in every home by checking related attributes (e.g., all homes have electricity, exterior materials, a zoning category).
The missingness therefore represents minor data entry gaps rather than absence of the feature.
To preserve the natural class distribution and avoid introducing artificial categories, we imputed these values using the mode — the most frequent category in the training data.


---

🔹 5. Small-Missing Numeric Features → Median

Some numerical attributes had only a few missing values (e.g., GarageYrBlt).
These are genuine recording gaps; using the median avoids distortion from outliers.


---

### Duplicates

As part of the data quality assessment, the dataset was examined for duplicate rows using both full-row duplication checks and ID-based checks.

No duplicate records were found in either the training or test dataset.

Because the dataset is already structurally clean and does not contain repeated entries, no duplicate removal step was required, ensuring the integrity and completeness of the dataset remains intact.


---

### 📊 Distribution Analysis Summary

The numerical features in the dataset show three major types of distributions:

1. Right-skewed distributions:
Many size-related features (e.g., LotArea, GrLivArea, TotalBsmtSF, 1stFlrSF) contain a long right tail because most homes are moderately sized while a few properties are extremely large.

2. Discrete / ordinal-like distributions:
Quality and condition ratings (e.g., OverallQual, OverallCond, KitchenQual) appear as step-wise or clustered values rather than smooth continuous variables, reflecting the scoring system used by assessors.

3. Zero-inflated distributions:
Features such as PoolArea, MiscVal, 3SsnPorch, and LowQualFinSF have a large number of zeros because most houses simply do not have these amenities.

<img width="1990" height="4989" alt="image" src="https://github.com/user-attachments/assets/0aef7bc0-eca9-4a77-b96c-df131a14fab3" />

The distribution analysis helped identify the transformations: 
* Ordinal Encoding for Ordered Quality Features- The distribution plots showed clear ranking patterns in quality features. Since these features represent increasing levels of quality, treating them as simple text categories would lose meaningful ordering.
* Outlier Removal: Skewed distributions revealed a few extreme outlier homes.


---

### Outlier Handling

I first generated boxplots for major numerical features.

<img width="1790" height="2989" alt="image" src="https://github.com/user-attachments/assets/684df531-608a-400d-9a82-5d006cec90a1" />

The visual inspection showed:
* Extremely large houses
* Very large lots and basements
* Rarely high porch/deck square footage
* Few observations greatly exceeding the upper whiskers

These points appeared as distinct outliers, signalling potential data-entry issues or extremely rare property types.

We used three structured steps to treat outliers while keeping the dataset realistic:
A. Domain-Based Removal

A small number of homes had:
> Very large living area but unusually low sale price, e.g., GrLivArea > 4500 combined with very low SalePrice. Such cases are likely inaccurate records or extremely unusual homes, and removing them avoids distortion.

B. IQR Filtering (Interquartile Range)

For features with obvious extreme tails, we applied the standard IQR rule:

lower = Q1 - 1.5 × IQR  
upper = Q3 + 1.5 × IQR

Values below lower or above upper were removed from train

For test, values were clipped within the same bounds

C. Winsorization (Capping)
Some features naturally contain many zeros but occasionally spike to very high values. Instead of removing rows, we capped values at the 99th percentile, keeping unusual values within a reasonable range while preserving their relative ordering.

Boxplots After Outlier Cleaning:

<img width="1989" height="4989" alt="image" src="https://github.com/user-attachments/assets/3f782d6a-ca06-4b9d-accb-74a7b6b22942" />

This visual confirmation shows that the outlier treatment was successful and helped stabilise the dataset.


---

### Biveriate analysis

To understand what drives property prices, I examined how each feature relates to the target variable (SalePrice). This helps reveal which characteristics contribute the most to value, how strong each relationship is, and whether the relationship behaves in a linear or nonlinear manner.

<img width="1989" height="4990" alt="image" src="https://github.com/user-attachments/assets/fe419bfb-2ea5-4784-a3d8-e183772b7ed2" />

1. Strong Linear Relationships
Several continuous variables showed clear linear or near-linear relationships with SalePrice:
* GrLivArea (Above-ground living area) – Larger homes consistently sell for more, forming an upward linear trend.
* TotalBsmtSF (Basement area) – More basement space correlates with higher price.
* GarageCars / GarageArea – Homes with larger garages show steady price increases.
* OverallQual (Overall material & finish quality) – Each increase in quality level results in a noticeable price jump, showing a strong monotonic trend.
- These patterns indicate that size and quality features influence price in a predictable, proportional way.

2. Moderate Relationships
Some features had weaker but still meaningful positive trends:
* YearBuilt – Newer homes tend to be more expensive.
* YearRemodAdd – Recently renovated homes show moderately higher values.
* FullBath and Fireplaces – More amenities generally increase price but with more variability.
- These variables contribute pricing signals but with more scatter around the trend.

3. Weak Relationships
Certain features showed little or inconsistent relationship with SalePrice:
* PoolArea, MiscVal, 3SsnPorch, LowQualFinSF, ScreenPorch
- These features are either rare, zero-heavy, or only add value for a small number of homes, causing flat or noisy bivariate patterns.


---

### Correlation matrix

To better understand how numerical features relate both to each other and to the target variable (SalePrice), a correlation matrix was generated. This allowed me to identify which variables are strongly associated with housing prices and which variables may be redundant or highly interrelated.

<img width="1205" height="928" alt="image" src="https://github.com/user-attachments/assets/3938ba06-6493-47ab-a691-ef5c377aed72" />

1. Strong Positive Correlations With SalePrice

Several features showed high positive correlation with SalePrice, meaning prices tend to increase as these values increase:
* OverallQual – strongest correlation; higher quality materials and finishes lead to higher prices.
* GrLivArea – larger above-ground living area strongly increases property value.
* GarageCars / GarageArea – more garage capacity is associated with higher prices.
* TotalBsmtSF – bigger basement area contributes significantly to value.
* YearBuilt & YearRemodAdd – newer or recently renovated homes tend to be priced higher.
- These variables provide strong signals and are important for understanding what drives price variation.

2. Weak Correlations
Some variables displayed near-zero or slightly negative correlation with SalePrice:
* EnclosedPorch, MiscVal, LowQualFinSF, 3SsnPorch, PoolArea (mostly zeros)
- These features add little pricing information due to their rarity or limited contribution to value.

3. Multicollinearity Patterns
The matrix also revealed several features that are highly correlated with each other, for example:
* GarageCars and GarageArea
* GrLivArea and 1stFlrSF
* TotalBsmtSF and 1stFlrSF
* YearBuilt and YearRemodAdd
- This means some variables capture similar underlying information (e.g., home size), and may need feature engineering.


---

### 📊Multivariate analysis

After exploring numerical and categorical features individually, a multivariate analysis was performed to understand how multiple features interact together to influence housing characteristics.

<img width="2456" height="2538" alt="image" src="https://github.com/user-attachments/assets/e4fb3b3a-9031-4312-8887-07cbf3d7d499" />


The pairplot highlights how the most important numerical features interact with each other and with SalePrice. Strong upward trends appear between SalePrice and features like OverallQual, GrLivArea, GarageCars, and TotalBsmtSF, confirming clear linear and monotonic relationships. Several variables form step-like clusters (e.g., GarageCars, FullBath), showing why ordinal encoding works well.

Overall, the pairplot shows mostly linear patterns with no major nonlinear structures, helping justify a linear modeling approach after proper preprocessing.


---

### 🧩Feature Engineering

To improve signal extraction and strengthen relationships between predictors and SalePrice, several domain-informed features were engineered:

| Feature | Description |
| :--- | :--- |
| `TotalSF` | Total square footage (basement + floors) |
| `Total_Bathrooms` | Combined full/half baths (above + basement) |
| `HouseAge` | Years since built |
| `RemodAge` | Years since remodel |
| `OverallQualArea` | Quality score (`OverallQual` $\times$ `TotalSF`) |
| `QualityScore` | Overall quality $\times$ condition |
| `TotalPorch` | Combined porch areas |

These engineered features reduce noise and strengthen linear relationships—making patterns more predictable and improving model performance.


---

### Encoding Strategy

To prepare the dataset for modeling, two encoding strategies were applied depending on the nature of each feature:
* **Ordinal Encoding:** Applied to ordered quality variables (e.g., `ExterQual`, `KitchenQual`, `BsmtQual`, `FireplaceQu`).
* **One-Hot Encoding:** Applied to nominal categorical variables (e.g., `Neighborhood`, `MSZoning`, `HouseStyle`, `Exterior1st`). Train/Test alignment was automatically handled to ensure consistent feature sets.


---

### Train-test split

After preprocessing and encoding, the dataset was split into training and validation sets to evaluate model performance before generating predictions for the unseen test set.
I used an 80/20 split, where:
* 80% (training set)
* 20% (validation set)

## Model Development

Several linear and tree-based models were trained on the original SalePrice values, using the cleaned and feature-engineered dataset. A 5-fold cross-validation strategy was used to compare performance.

| Model	| Mean R² |
| :--- | :--- |
| Linear Regression	| 0.91119 |
| Lasso Regression | 0.91116 |
| Ridge Regression | 0.91072 |
| ElasticNet |	0.91139 |
| Random Forest	| 0.89988 |
| Gradient Boosting |	0.91104 |
| XGBoost |	0.91669 (Highest R²) |

### Why Ridge Regression?
Although XGBoost achieved the highest R², Ridge Regression was chosen because it offered the best balance between accuracy, simplicity, and stability. Ridge handles high-dimensional one-hot encoded data very well, reduces multicollinearity, and provides more consistent and interpretable results across folds. It also runs faster and performs nearly as well as the top tree-based models, making it the most practical final choice.

### Hyperparameter Tuning (GridSearchCV)

To improve model performance, I performed hyperparameter tuning specifically for Ridge Regression, since it was the most promising linear model during baseline evaluation.

I used a GridSearchCV over a small, efficient search space:
* param_grid = {"alpha": [0.1, 1, 10, 50, 100]}

The tuning process used 5-fold cross-validation with R² as the scoring metric to ensure the model performed consistently across different subsets of the data.

The search identified:
* Best alpha: 10
* Best cross-validated R²: 0.9215

After selecting this parameter, the tuned model was evaluated on a separate validation set:
* Validation R²: 0.9053
* Validation RMSE: 18,722.50


---

## 🧪 Model Diagnostics

### ✔ Validation Curves
The validation curve evaluates Ridge Regression across a range of alpha (regularization) values.
* Low alpha → overfitting (high train performance, lower validation).
* Medium alpha (≈10) → best performance (highest validation R², lowest RMSE).
* High alpha → underfitting (both train and validation performance drop). 

<img width="1389" height="489" alt="image" src="https://github.com/user-attachments/assets/435f5082-7eca-4d60-9b1e-d04d8a6b750b" />

### ✔ Learning Curve
The learning curve shows how model performance changes as training data increases.

Training and validation curves gradually converge → healthy bias–variance balance.

Validation error decreases with more data → model learns meaningful patterns.

No large gap → no major overfitting.

These curves confirm that Ridge Regression generalizes well and benefits from more data

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/ee9cb7bf-c9f2-407e-ae20-516a94c820f7" />


---

## 📉Error Analysis

To evaluate the performance and reliability of the final Ridge Regression model, a detailed error analysis was carried out using the validation set. This step helps identify how well the model generalizes, where it struggles, and whether the errors follow reasonable statistical behavior.

Performance Metrics

| MAE	| 15,627.57	| Average absolute difference between predictions and actual prices. |
| :--- | :--- | :--- |
| RMSE	| 21,347.87	| Penalizes larger errors more strongly. |
| R²	| 0.8798	| Model explains ~88% of the variance in SalePrice. |
| MAPE	| 9.83%	| On average, predictions deviate by ~9.8% from actual values. |
| Residual Mean	| 1,804.67	| Slight positive bias (minor overprediction). | 
| Residual Std	| 21,324.30 |	Dispersion of errors. | 

The metrics indicate good overall performance, with relatively low error values considering the wide price range of houses. The R² score shows the model captures most of the signal, and the MAPE below 10% is strong for housing price prediction.

**Residual distribution and Residuals vs Predicted**

<img width="1187" height="490" alt="image" src="https://github.com/user-attachments/assets/65d21c52-5eb8-4b20-8247-ed50a326725b" />

Residual Distribution
- The residuals (Actual − Predicted) are centered around zero and follow an approximately normal shape. This indicates that the model’s errors are mostly random rather than systematic, meaning the model does not consistently over- or under-predict.

Residuals vs Predicted
- Residuals are scattered randomly around the zero line with no clear trend. This shows that the model maintains fairly consistent error across different price ranges and does not suffer from major heteroscedasticity. Larger spread at very high predicted prices is expected due to higher variability in expensive homes.

Actual vs Predicted

<img width="577" height="547" alt="image" src="https://github.com/user-attachments/assets/f80c24b9-abaf-403b-a1b7-d1f8c44bde46" />

This scatterplot compares true house prices to the model’s predictions. Most points lie close to the diagonal line, meaning the model predicts values that are very close to the actual prices. The spread widens slightly for higher-priced homes, which is expected, but there is no clear pattern of consistent over- or under-prediction. Overall, the plot shows that the Ridge model captures the price trend well and produces stable, reasonable predictions.


---

## Model Explainability (SHAP)

**SHAP (SHapley Additive exPlanations)** was used to interpret the model's predictions and understand feature contributions. 

The bar plot ranks the top features by their average impact on predicted sale price. Features like OverallQualArea, OverallQual, and TotalSF have the strongest influence on the model.

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/fd8a93da-a180-44cd-8dde-da41d4673cef" />

The dot plot shows how each feature affects predictions.

<img width="778" height="940" alt="image" src="https://github.com/user-attachments/assets/b5438c0c-6c1e-4f37-94eb-c81f8b56cef5" />

Overall, SHAP confirms that home quality and total living area are the biggest drivers of price, and that the model behaves consistently with real-estate intuition.


---

## Final Submission

The tuned Ridge model was retrained on the full training dataset and used to generate the final predictions.

* **Kaggle Score:** **0.13820** (RMSE)

<img width="1033" height="318" alt="image" src="https://github.com/user-attachments/assets/8e660fb3-571c-423e-9dcc-e81e3c61f9f5" />


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

## 🏁Conclusion

This project delivered a complete end-to-end House Price Prediction workflow—from data cleaning and feature engineering to model training, evaluation, and deployment. After comparing multiple models, Ridge Regression was selected for its stable performance and robustness to multicollinearity.

The model achieved strong validation metrics and was further explained using SHAP to confirm the most influential features. Finally, the solution was deployed using Streamlit, providing an easy-to-use interface where users can input home features and receive instant price predictions.


---

## Future works
* Refine Ridge regularization by exploring a wider alpha range or adaptive regularization to further reduce prediction error.
* Investigate remaining error patterns using deeper SHAP analysis to identify features causing consistent under/over-estimation.
* Incorporate neighborhood-level external data (e.g., crime rates, school scores, accessibility) to strengthen price prediction signals.
* Test alternative linear approaches, such as ElasticNet with tuned L1/L2 balance, to see if sparsity improves model generalization.
* Enhance Streamlit app usability by grouping features more intuitively and adding optional “advanced inputs” for power users.

---
