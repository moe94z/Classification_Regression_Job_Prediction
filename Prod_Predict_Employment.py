import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import pickle

from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, roc_curve, auc, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

def send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def Pipeline(filepath, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password):
    try:
        # Data Loading and Initial Review
        df = pd.read_excel(filepath)
        print("The shape of the data:\n", df.shape)
        print("\nThe first 5 rows are:\n", df.head())
        print("\nThe last 5 rows are:\n", df.tail())
        print("\nThe column names are:\n", df.columns)
        
        # Data Cleaning and Preprocessing
        df["placed"] = df["placed"].replace({0: "not_placed", 1: "placed"})
        df["pathrise_status"] = df["pathrise_status"].replace(
            ["Withdrawn", "Withdrawn (Trial)", "MIA", "Deferred"], "not_enrolled"
        ).replace(
            ["Placed", "Withdrawn (Failed)", "Active", "Closed Lost", "Break"], "enrolled"
        )
        
        # Remove rows with missing cohort_tag
        df.drop(df[df["cohort_tag"] == "OCT21A"].index, inplace=True)
        
        # Handle missing values
        df["professional_experience"] = df["professional_experience"].replace(
            ["Less than one year"], 6).replace(
            ["1-2 years"], 18).replace(
            ["3-4 years"], 42).replace(
            ["5+ years"], 60)
        
        # Impute missing values with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Impute missing values with mode for categorical columns
        categorical_cols = df.select_dtypes(include=[object]).columns
        df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))
        
        df["log_days"] = df["days_program"].apply(lambda x: np.log(abs(x)) if x > 0 else 0)
        
        # Encode categorical variables
        df = pd.get_dummies(df, columns=[
            "primary_track", "cohort_tag", "employment_status ", "education", 
            "length_of_job_search", "professional_experience", 
            "work_authorization_status", "gender", "race"
        ])
        
        # Feature Selection for Classification
        X = df.drop(columns=["placed", "id", "days_program", "log_days"])
        y = df["placed"].replace({"not_placed": 0, "placed": 1})
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Logistic Regression Model
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        y_pred = logistic_model.predict(X_test)
        f1_score_value = f1_score(y_test, y_pred)
        print(f'Logistic Regression F1 Score: {f1_score_value:.4f}')
        
        # ROC Curve and AUC
        fpr, tpr, _ = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Logistic Regression')
        plt.legend(loc="lower right")
        plt.show()
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': logistic_model.coef_[0]
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance.head(10))
        
        # Regression Model
        X_reg = df.drop(columns=["placed", "id", "days_program"])
        y_reg = df["log_days"]
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        # Linear Regression Model
        linear_model = LinearRegression()
        linear_model.fit(X_train_reg, y_train_reg)
        y_pred_reg = linear_model.predict(X_test_reg)
        mse_linear = mean_squared_error(y_test_reg, y_pred_reg)
        print(f'Linear Regression MSE: {mse_linear:.4f}')
        
        # XGBoost Regression Model
        xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_reg.fit(X_train_reg, y_train_reg)
        y_pred_xgb = xgb_reg.predict(X_test_reg)
        mse_xgb = mean_squared_error(y_test_reg, y_pred_xgb)
        print(f'XGBoost Regression MSE: {mse_xgb:.4f}')
        
        # Feature Importance for Regression Model
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(xgb_reg, ax=plt.gca())
        plt.show()
        
        # Send email notification with results
        subject = "Pipeline Execution Results"
        body = (f"Logistic Regression F1 Score: {f1_score_value:.4f}\n"
                f"Linear Regression MSE: {mse_linear:.4f}\n"
                f"XGBoost Regression MSE: {mse_xgb:.4f}")
        send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Send email notification with error
        subject = "Pipeline Execution Error"
        body = f"An error occurred: {e}"
        send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password)

# SMTP and email configuration
sender_email = "prod_run@dummy_email.com"
receiver_email = "redacted"
smtp_server = "redacted"
smtp_port = 25
smtp_user = "redacted"
smtp_password = "redacted"

# Run the Pipeline function with the file path to your data and email configurations
Pipeline("Data_Pathrise.xlsx", sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password)
