# AI-Powered Used Car Valuation Assistant

Live Demo: https://used-car-ai.streamlit.app  
GitHub Repository: https://github.com/umarmohdcr7-spec/data-analysis-agent

## Overview
An end-to-end machine learning web application that predicts fair market prices for used vehicles and evaluates whether a listing is overpriced, underpriced, or fairly valued.

## Key Features
- Fair market price prediction
- Overpriced / underpriced detection
- Confidence price range
- Valuation score
- Explainable valuation summary
- Market insights dashboard
- Downloadable valuation report
- Streamlit Cloud deployment

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git / GitHub

## Machine Learning Workflow
1. Data cleaning and preprocessing
2. Exploratory Data Analysis
3. Feature engineering
4. Log-price transformation
5. Linear Regression baseline
6. Random Forest model
7. Hyperparameter tuning
8. Model deployment with Streamlit

## Model Performance
- Linear Regression: R² ≈ 0.90
- Random Forest: R² ≈ 0.94

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
