# Real Estate Market Dynamics: Predicting Property Prices and Analyzing Market Trends

## **Project Overview**
This project aims to analyze real estate market behavior and predict property sale prices using machine learning and deep learning models.  
By leveraging transaction data from **2009–2022**, the project seeks to identify key factors that drive price variation across localities and property types.  
The outcome includes both predictive models and visual insights to assist buyers, sellers, and investors in making data-driven decisions.

---

## **Dataset**
**Source:** [OpenDataBay - Real Estate Transactions](https://www.opendatabay.com/data/financial/e67485f2-6d3c-4a3b-93a6-4b76ca07e5fb?utm)

**Features Include:**
- Date, Year, Locality, Property Type, Residential Indicator  
- Estimated Value, Sale Price, Carpet Area, Property Tax Rate  
- Number of Rooms, Bathrooms, Face Orientation  

---

## **Problem Statement**
Accurately estimating property prices is one of the most critical challenges in the real estate market.  
Prices vary significantly across localities and are influenced by multiple factors such as property type, area, and tax rate.  
Despite data availability, many valuations rely on manual estimates, leading to pricing errors and missed opportunities.  

This project uses **machine learning** and **deep learning** to build predictive models that estimate sale prices and uncover market trends across regions and years.

---

## **Project Objectives**
- Predict property sale prices using regression techniques.  
- Identify market trends and key factors influencing property values.  
- Compare traditional ML algorithms with a Deep Neural Network (DNN).  
- Visualize insights via charts and dashboards for decision-making.

---

## **Methodology**

This project follows a structured seven-phase workflow combining data preprocessing, exploratory analysis, feature engineering, and predictive modeling.

---

### **1️⃣ Data Understanding**

The first step is to explore and familiarize ourselves with the dataset:
- Identify feature types (numeric, categorical, date).
- Check for missing values, inconsistencies, and outliers.
- Understand data ranges and patterns in pricing and property attributes.

**Goal:** Build a foundational understanding of what the data represents and identify necessary cleaning or transformations.

---

### **2️⃣ Data Cleaning & Preprocessing**

Data cleaning ensures consistency and reliability.

**Steps:**
- **Handle Missing Values:**  
  - Fill categorical features (e.g., *Locality*) with “Unknown.”  
  - Impute numerical columns (e.g., *Estimated Value*, *Carpet Area*) with the **median per category** (like Property Type or Locality).  

- **Outlier Handling (Winsorization):**  
  - Cap extreme values in *Sale Price*, *Estimated Value*, and *Carpet Area* based on IQR thresholds to reduce skew and distortion.  

- **Feature Conversion:**  
  - Convert `Date` to datetime.  
  - Derive new time-based features — `Year`, `Month`, `Quarter`.

- **Encoding & Normalization:**  
  - Encode categorical variables.  
  - Normalize numeric columns for model stability.

**Goal:** Produce a clean, ready-to-model dataset.

---

### **3️⃣ Exploratory Data Analysis (EDA)**

EDA helps uncover trends and relationships among variables.

**Key Visualizations:**
- Sale Price distribution (with and without log transform).  
- Average Sale Price by Locality and Property Type.  
- Yearly Price Trends (2009–2022).  
- Correlation Heatmap of numeric variables.  
- Relationship between features (e.g., rooms vs price, area vs value).  

**Goal:** Extract insights and understand data behavior before modeling.

---

### **4️⃣ Feature Engineering**

Create meaningful and model-ready features.

**Steps:**
- Generate derived metrics (e.g., price per square meter, room-to-bathroom ratio).  
- Apply log transformations on skewed variables.  
- Identify and retain top features using feature importance scores.  

**Goal:** Enhance the dataset’s predictive strength and interpretability.

---

### **5️⃣ Model Building**

Train multiple models and compare their performance.

#### **a. Linear Regression**
- Acts as the baseline model.  
- Assumes a linear relationship between features and target (Sale Price).  
- Useful for understanding directional impact of variables.

#### **b. Random Forest Regressor**
- An ensemble of decision trees that reduces overfitting.  
- Handles non-linear relationships and feature interactions effectively.  
- Offers built-in feature importance for interpretability.

#### **c. XGBoost Regressor**
- A powerful gradient boosting algorithm.  
- Builds trees sequentially to minimize error efficiently.  
- Known for superior predictive accuracy and flexibility.

#### **d. Deep Neural Network (DNN) — Keras Sequential API**
- A **Deep Learning** model using TensorFlow’s **Keras Sequential API** for regression.  
- Selected for its simplicity and layer-by-layer design.

**Network Design:**
- Input: scaled numeric + encoded categorical variables.  
- Hidden Layers: multiple Dense layers with **ReLU** activation.  
- Regularization: Dropout + Batch Normalization.  
- Output: single neuron (linear activation) for continuous prediction.

**Training Setup:**
- Optimizer: Adam  
- Loss Function: Mean Squared Error (MSE)  
- Metrics: Mean Absolute Error (MAE)  
- Training Techniques: early stopping, learning rate scheduling.

**Goal:** Learn complex nonlinear dependencies and benchmark deep learning performance against traditional models.

---

### **6️⃣ Model Evaluation & Interpretation**

Each model will be assessed using the following metrics:

| Metric | Description |
|---------|--------------|
| **MAE** | Average magnitude of prediction errors |
| **RMSE** | Penalizes large errors more heavily |
| **R²** | Explains how much variance is captured by the model |

**Interpretation Tools:**
- Predicted vs Actual Price plots.  
- Feature Importance visualization (for RF and XGBoost).  
- Learning Curves (for DNN training performance).

**Goal:** Evaluate and interpret which model performs best and why.

---

### **7️⃣ Reporting & Insights**

**Deliverables:**
- Data cleaning summary (missing values, outlier handling).  
- EDA findings and statistical highlights.  
- Model performance comparison table.  
- Business insights:
  - Factors driving higher prices.
  - Localities with rapid appreciation.
  - Characteristics associated with high property value.

**Goal:** Provide actionable, data-driven insights for stakeholders in real estate investment and policy-making.

---

## **Expected Outcome**
- A regression-based framework for accurate price prediction.  
- Visual and statistical evidence of market behavior.  
- Comparative model insights showcasing where machine learning and deep learning excel.  
- A replicable data science pipeline applicable to other market datasets.

---

## **Technologies & Tools**
- **Languages:** Python  
- **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, TensorFlow (Keras)  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Environment:** Jupyter Notebook / VS Code  

---

## **Learning Goals**
- Gain hands-on experience with **real-world regression modeling**.  
- Learn **deep learning model design** using the **Keras Sequential API**.  
- Understand the impact of **feature engineering and regularization** in predictive modeling.  
- Strengthen **data storytelling and dashboarding** through visualization.

---

*By combining traditional ML and deep learning, this project aims to reveal how property attributes and locality characteristics shape real estate values — and how predictive analytics can guide smarter investment decisions.*
