# Customer Churn Analysis and Prediction Dashboard

## Overview

This project analyzes customer churn behavior in a telecommunications dataset and applies machine learning models to predict whether a customer is likely to leave the service.

The application is built using Streamlit and provides an interactive interface for dataset exploration, model evaluation, feature importance analysis, and real-time churn prediction.

## Features

* Exploratory Data Analysis (EDA) on customer churn data
* Data preprocessing using Scikit-Learn Pipelines
* Handling missing values and categorical encoding
* Comparison of multiple machine learning models:

  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier
* Manual hyperparameter tuning
* Confusion Matrix visualization
* Feature Importance analysis for tree-based models
* Real-time prediction using custom user inputs

## Dataset

Dataset: Telco Customer Churn Dataset

The dataset contains customer demographic information, subscription details, service usage, billing information, and churn status.

Target Variable:

* Churn (Yes / No)

## Machine Learning Workflow

1. Data Cleaning
2. Missing Value Treatment
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. Prediction and Deployment

## Models Evaluated

| Model               | Purpose                       |
| ------------------- | ----------------------------- |
| Logistic Regression | Baseline Classification Model |
| Decision Tree       | Non-linear Classification     |
| Random Forest       | Ensemble Learning             |

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* Seaborn
* Plotly
* Joblib

## Running Locally

Clone the repository:

```bash
git clone https://github.com/deepakdkay07/customer_churn_analysis.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Live Demo

Streamlit Application:

https://dkay07.streamlit.app/

## Future Improvements

* Cross-validation based model comparison
* Automated hyperparameter optimization using RandomizedSearchCV
* Additional evaluation metrics such as Precision, Recall, F1-Score, and ROC-AUC
* Integration with SQL-based data pipelines
* Deployment of advanced gradient boosting models such as XGBoost

Learning Data Science, Machine Learning, SQL, and AI Engineering
