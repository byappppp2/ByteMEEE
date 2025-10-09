import os
import numpy as np
import pandas as pd
from textwrap import dedent

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# OpenAI Responses API
from openai import OpenAI


NROWS = 40_000                      # fixed at 40k
ACCOUNTS = "datasets/HI-Medium_accounts.csv"
TRANSACTIONS = "datasets/HI-Medium_Trans.csv"
RANDOM_SEED = 42

#load in the databases
accounts = pd.read_csv(ACCOUNTS) #print(accounts.head())
transactions = pd.read_csv(TRANSACTIONS) #print(transactions.head())

#rename the double header
transactions.columns = ['Timestamp', 'From Bank', 'From Account', 'To Bank', 'To Account', 'Amount Received', 'Receiving Currency',
                        'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']

# Convert key columns to string
transactions["From Account"] = transactions["From Account"].astype(str)
transactions["To Account"] = transactions["To Account"].astype(str)
accounts["Account Number"] = accounts["Account Number"].astype(str)

#create 2 dataframes for sending / receiving accounts
accounts_from = accounts.rename(columns={
    "Bank Name": "From Bank Name",
    "Bank ID": "From Bank ID",
    "Account Number": "From Account",
    "Entity ID": "From Entity ID",
    "Entity Name": "From Entity Name"
})

accounts_to = accounts.rename(columns={
    "Bank Name": "To Bank Name",
    "Bank ID": "To Bank ID",
    "Account Number": "To Account",
    "Entity ID": "To Entity ID",
    "Entity Name": "To Entity Name"
})

#merge onto transactions
df = transactions.merge(accounts_from, on="From Account",how="left")
df = df.merge(accounts_to,on="To Account", how="left")

#drop the duplicated columns
df = df.drop(columns=["To Bank ID", "From Bank ID"])
#print(df.head(10))

#split the correct outcomes away from the dataset
y = df["Is Laundering"].astype(int)
x_df = df.drop(columns=["Is Laundering"])

#encode the values that are not numeric
categorical_cols = [col for col in x_df.columns if col not in ["Amount Received", "Amount Paid"]]
for col in categorical_cols:
    x_df[col] = LabelEncoder().fit_transform(x_df[col].astype(str))
#print(x_df.dtypes)

#standardisation
scaler = StandardScaler().fit(x_df)
x_scaled = scaler.transform(x_df)

contamination = float(np.clip(y.mean(), 1e-6, 0.5))

#train the isolation forest
iforest = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
iforest.fit(x_scaled)

# Get Predictions
y_pred = (iforest.predict(x_scaled) == -1).astype(int)

print(len(y_pred), len(transactions), len(y))

assert len(y_pred) == len(y), "Length mismatch—recheck inputs."

# Evaluation
acc  = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec  = recall_score(y, y_pred, zero_division=0)
f1   = f1_score(y, y_pred, zero_division=0)
cm   = confusion_matrix(y, y_pred, labels=[1,0])

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion matrix [rows=Actual 1,0; cols=Pred 1,0]:\n", cm)

# Visualisation




























