# Real Estate Market Dynamics: Predicting Property Prices and Analyzing Market Trends

## Project Overview
Accurately estimating property prices is a critical challenge in the real estate market. Prices vary across localities and are influenced by multiple factors such as property type, residential style, carpet area, property tax rate, and year of sale.  

This project leverages machine learning and deep learning techniques to predict property sale prices and analyze market trends from 2009 to 2022.

## Project Objectives
- Predict property sale prices using features like `carpet_area`, `num_rooms`, `num_bathrooms`, `property_tax_rate`, and `locality`.  
- Analyze market trends over time and across regions.  
- Compare traditional machine learning algorithms with a Deep Neural Network (DNN).  
- Visualize insights through an interactive dashboard for buyers, sellers, and investors.

## Dataset
- **Source**: [OpenDataBay - Real Estate Market Dynamics](https://www.opendatabay.com/data/financial/e67485f2-6d3c-4a3b-93a6-4b76ca07e5fb?utm)  
- **Size**: ~10,000 property transactions  
- **Features**:  
  - `Date`, `Year`, `Locality`  
  - `Estimated Value`, `Sale Price`  
  - `Property` type, `Residential` type  
  - `num_rooms`, `num_bathrooms`  
  - `carpet_area`, `property_tax_rate`, `Face`

## Project Steps

### 1. Data Understanding
- Load and inspect the dataset.
- Identify numeric and categorical features.
- Check for missing values, duplicates, and outliers.

### 2. Data Cleaning & Preprocessing
- Handle missing values.
- Convert `Date` to datetime and extract `Year`, `Month`, `Quarter`.
- Encode categorical variables: `Locality`, `Property`, `Residential`, `Face`.
- Normalize numeric features for deep learning models.

### 3. Exploratory Data Analysis (EDA)
- Distribution plots for `Sale Price` and `Estimated Value`.
- Boxplots and trend plots for property types and localities.
- Correlation analysis to identify relationships between numeric features.

### 4. Feature Engineering
- Create derived features such as `price_per_sqft = Sale Price / carpet_area`.
- Standardize numeric features.
- Select important features using feature importance scores.

### 5. Model Building
- **Traditional ML Models**: Linear Regression, Random Forest Regressor, XGBoost Regressor.
- **Deep Learning Model**: Feedforward Neural Network (DNN) using Keras.
- Evaluate models using MAE, RMSE, and R² score.

### 6. Evaluation & Interpretation
- Compare all models’ metrics side by side.
- Visualize predicted vs actual prices.
- Analyze feature importance to understand price drivers.

### 7. Visualization & Dashboard
- Build an interactive dashboard using Tableau.
- Features:
  - Average sale price by locality and property type.
  - Historical trend analysis.
  - Input sliders for feature-based price predictions.

### 8. Reporting & Insights
- Summarize dataset overview and cleaning steps.
- Present model performance comparison.
- Highlight business insights (e.g., key price-driving factors).
- Provide visual market trend analysis.

## Deliverables
- Cleaned and preprocessed dataset.
- EDA visualizations and correlation plots.
- Machine learning and deep learning models.
- Performance evaluation metrics.
- Feature importance analysis.
- Optional interactive dashboard.
- Final project report or presentation slides.

## Tools & Technologies
- **Data Handling**: Python, Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn, Plotly, Tableau / Power BI  
- **Machine Learning**: scikit-learn, XGBoost  
- **Deep Learning**: TensorFlow / Keras  
- **Dashboard**: Tableau  
- **Environment**: VS Code
