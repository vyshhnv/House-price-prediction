
🏠 House Price Prediction using Machine Learning

📌 Overview

This project is an end-to-end Machine Learning application that predicts house prices using structured housing data.
The workflow includes data preprocessing, training multiple regression models, evaluating their performance, selecting the best model, and deploying it using Streamlit.
________________________________________
Project Files

House Price Prediction.csv   - Dataset  
analysis.ipynb               - Model training, evaluation, and tuning  
app.py                       - Streamlit application  
README.md                    - Documentation  
________________________________________
Dataset

The dataset contains housing data with the following features:

•	Area

•	Bedrooms

•	Bathrooms

•	Floors

•	YearBuilt

•	Location (Downtown, Suburban, Rural)

•	Condition (Poor, Fair, Good, Excellent)

•	Garage (Yes / No)

Target: Price
________________________________________
Data Processing

•	Removed unnecessary columns

•	Scaled numerical features

•	Encoded categorical features

•	Applied the same preprocessing during training and prediction
________________________________________
Machine Learning Models Used

The following regression models were implemented and evaluated:

•	Linear Regression

•	Ridge Regression

•	Decision Tree Regressor

•	Random Forest Regressor

•	Gradient Boosting Regressor

Based on evaluation metrics, the Random Forest Regressor performed best and was selected for hyperparameter tuning and final deployment.
________________________________________
Deployment

The tuned model is deployed using Streamlit.

Run the application:
streamlit run app.py

The application allows users to input house details and receive a predicted price.
________________________________________
Tools Used

•	Python

•	Pandas, NumPy

•	Scikit-learn

•	Streamlit


