# 🎓 Student Dropout Early Warning System (EWS)
## Project Overview

This project predicts which university students are at risk of dropping out early in the semester. It uses students' academic activity and basic demographic information to calculate a risk score and classify each student as Low, Medium, or High risk, so advisors can focus on students who need attention before it's too late.

I built this as part of my AI & Data Science practice. It combines data preprocessing, feature engineering, machine learning, and a user-friendly Streamlit app.

## Dataset

xAPI-Edu-Data on Kaggle

## Key Features
Predicts student dropout risk
Calculates risk score and risk label (Low / Medium / High)
Shows the Top 20 high-risk students for advisors
Displays individual student details and top reasons for risk
Allows download of predictions as CSV
Built with Python, Pandas, XGBoost, and Streamlit

# Requirements
Dataset (from Kaggle link above)
Python
Streamlit and the libraries mentioned above

bash

pip install pandas xgboost streamlit
streamlit run app.py

*Thank you for checking out this project! Feel free to explore the code and share feedback.*

