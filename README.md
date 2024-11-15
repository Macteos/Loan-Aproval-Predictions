# Loan Approval Prediction

This project is a machine learning solution to predict loan approval status using various applicant-related features. The project involves data exploration, preprocessing, model training, and evaluation to determine the best model for accurate predictions. The dataset contains various features like applicant's age, income, home ownership, and credit history.

## Project Overview

The objective of this project is to build a machine learning model that can predict whether a loan application will be approved based on the applicant’s personal and financial information. The steps involved include:

1. **Data Loading and Exploration**: Understanding the dataset and its features through descriptive statistics and visualizations.
2. **Data Preprocessing**: Cleaning the data, handling missing values, and performing feature engineering to prepare it for modeling.
3. **Modeling**: Training different machine learning models (Logistic Regression, Decision Trees, Random Forests, etc.) and tuning them for better performance.
4. **Evaluation**: Comparing models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
5. **Conclusion and Future Work**: Summarizing results and suggesting improvements for future versions.

## Dataset

The dataset contains various features related to applicants applying for loans:

- **person_age**: Age of the applicant
- **person_income**: Income of the applicant
- **person_home_ownership**: Type of home ownership (e.g., rent, mortgage, etc.)
- **person_emp_length**: Length of employment in years
- **loan_intent**: Purpose of the loan (e.g., debt consolidation, home improvement)
- **loan_grade**: Loan grade (A to G, with A being the highest quality)
- **loan_amnt**: Loan amount
- **loan_int_rate**: Interest rate of the loan
- **loan_percent_income**: Percentage of income used for loan repayment
- **cb_person_default_on_file**: Whether the person has defaulted on a loan before
- **cb_person_cred_hist_length**: Length of the person's credit history
- **loan_status**: Target variable (approved or not approved)

## Project Structure

The project is organized as follows:

```bash
Loan-Approval-Predictions/
├── data/
│   ├── raw/                  # Raw data file (train.csv, test.csv)
│   ├── processed/             # Processed data after cleaning
├── notebooks/                 # Jupyter Notebooks for EDA and modeling
│   ├── 1_EDA.ipynb            # Exploratory Data Analysis
│   ├── 2_Preprocessing.ipynb  # Data Preprocessing and Feature Engineering
│   ├── 3_Modeling.ipynb       # Model Training and Selection
│   ├── 4_Evaluation.ipynb     # Model Evaluation and Metrics
│   ├── 5_Complete_Process.ipynb     # Model Evaluation and Metrics
├── src/                       # Python Scripts for Modular Code
│   ├── data_processing.py     # Data preprocessing functions
│   ├── model.py               # Model training and evaluation functions
├── results/                   # Visualizations and final output files
├── README.md                  # This file
├── requirements.txt           # List of dependencies
