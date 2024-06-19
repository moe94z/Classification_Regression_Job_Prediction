### Classification_Regression_Job_Prediction

#### HR Data Science Problem: Predict Employement based on the fellows creditionals.
*** Predictive Modeling,EDA, Visualization, RandomizedSearch Hypertuning a. two models 1. Classification: IF a fellow will be placed or not. 2. Regression: WHEN a fellow will be placed

Problem Statement
The journey to employment is not an easy one. We spend our time learning and developing skills to eventually get a job. The research conducted in the following notebook will focus on predicting if a fellow will be placed and the time it will take for a fellow to be placed. Throughout the notebook, I will derive insights around fellows and create two models to effectively predict if and when a fellow will be placed.

That being said, there are two models we are going to create in this case study:

Classification model: Whether or not the fellow will be placed.
Regression model: Predict the length that it will take a fellow to find placement.
The other questions that I will tackle in the research:

Overall placement, how many individuals in the program were placed?
Pathrise Placement, did pathrise have an impact?
What is the education of the fellows that were placed, did the education impact placement?
Duration until placement, how long does it take a fellow to be placed?



# Data Science Pipeline for Pathrise Placement Prediction

## Overview

This repository contains a Python script that implements a data processing and machine learning pipeline to predict the placement of fellows in the Pathrise program. The pipeline includes data loading, preprocessing, feature engineering, model training, evaluation, and email notifications.

## Features

- Data loading from an Excel file
- Data cleaning and preprocessing
- Handling of missing values
- Feature encoding using one-hot encoding
- Logistic Regression model for classification
- Linear Regression and XGBoost models for regression
- Evaluation metrics including F1 score, Mean Squared Error (MSE), and ROC curve
- Email notifications for pipeline execution results or errors

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scipy
- Scikit-learn
- XGBoost
- smtplib (for email notifications)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/moe94z/pathrise-placement-prediction.git
   cd pathrise-placement-prediction




![data distribution](https://user-images.githubusercontent.com/58402096/159050511-c731fc28-d703-4dec-9cc8-0532b985e8bc.png)
