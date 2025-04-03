# Data Science Finance - Capstone Project
- Client: Lending Club*
- Project: Data Cleaning, Exploratory Data Analysis, and Predictive Modeling on Loan Application Dataset

---
## Project Overview

Lending Club is seeking the expertise of a data science consultant
to perform comprehensive data cleaning, exploratory data
analysis (EDA), and predictive modeling on their loan application
dataset. The project will also explore the potential for deploying
a real-time scoring application. The primary objective is to prepare
the dataset for accurate analysis and modeling, understand the
key variables influencing loan approval, and recommend a
predictive model for classifying loan applications.

## Business Understanding:

The Lending Club specialize is lending loans, and would like help in making decisions to approve or reject loan applications based on applicant's profile, and minimise the risk of financial losses e.g., losing money via loans that don't get payed back, hence the below bullet points.

- If a applicant is likely to repay the loan, 
    - **Not approving the loan =** Loss of business to the company
    - **Approving =** Company generates revenue i.e., financial gain !

- If a applicant is NOT likely to repay the loan, i.e., likely to a loan default
    - **Approving =** Company leads to financial losses
    - **Not approving =** Company has reduced the risk of potential financial losses

---

## Dataset Description

The dataset consists of loan application records stored in

A CSV file 
club-2007-2020Q3/Loan_status_2007-2020Q3-100ksample.csv

The dataset contains various attributes such as applicant
information, loan details, financial metrics, and application
status

A data dictionary is provided LCDataDictionary.xlsx


## Objective
- Since the dataset does NOT have Loan *Accepted* or *Rejected*, we can utilise the `loan_status` field and determine which loans are most likely going to **DEFAULT**.
  
    ---
    - **Defaulted Loan:** A loan is considered in default when the borrower fails to make the required payments as agreed in the loan contract. Default typically occurs after missing several payments (usually 90 to 180 days, depending on the type of loan and lender policies).*
    - **A loan is charged off** when the lender writes off the loan as a bad debt on their financial statements, recognizing it as a loss. This typically happens after the loan has been in default for a significant period, often around 180 days.*


- We will explore **Loan_Default** criteria in section **1a**

---

### **Notebook Structure:**
0. Import/Ingest Lending Club Data

1. Data Preparation & Cleaning
    - 1a. Data Preparation *(selecting features)*
    - 1b. Feature Creation / Engineering
    - 1c. Data Cleaning

2. Exploratory Data Analysis (EDA)

3. Build Predictive Model for Loan Applications Approvals
    - 3a. Data Preparation for ML models
    - 3b. Base Model
    - 3c. Challenger Model
        - Feature Selection via Recursive Feature Elimination (RFE)
        - GridSearchCV to find best parameters
        - Finetune parameters of model to not overfit

4. Model Evaluations & Comparisons

5. Save Models for Deployment

**Insights & Conclusions**

