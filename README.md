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
- apache-airflow (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/moe94z/pathrise-placement-prediction.git
   cd pathrise-placement-prediction

2. Install the required Python packages using requirements.txt:
   ```bash
    pip install -r requirements.txt

## Execution
Python3 Predict_Employment.py > errors.log & 

Also can leverage Airflow to schedule job for production if necessary, creating a dag for the job is not complicated due to its low dependences 

##Airflow Dag dummy script

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'pathrise_pipeline',
    default_args=default_args,
    description='A simple pipeline for Pathrise placement prediction',
    schedule_interval=timedelta(days=1),
)

t1 = BashOperator(
    task_id='run_pipeline',
    bash_command='python3 local/environment/prod/Predict_Employment.py > local/environment/prod/errors.log',
    dag=dag,
)

t1

##Launch Airflow webserver after init db
airflow webserver -p 8080
airflow scheduler


