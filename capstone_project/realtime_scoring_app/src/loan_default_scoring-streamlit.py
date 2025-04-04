

api_url = "http://127.0.0.1:8000"

import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go
import numpy as np
import pickle
import pandas as pd
import datetime
import os


st.set_page_config(layout="centered")
css = '''
<style>
        .stMainBlockContainer {
            max-width:900rem;
        }
    </style>
'''
st.markdown(css, unsafe_allow_html=True)

# set datetime string
str_dt = datetime.datetime.now().strftime('%Y-%m-%d') ## String on DateTime in format YYYYMMDD

# Create figure and add one scatter trace
fig = go.Figure()

# Title
st.title("Lending Club - Loan Default Scoring App")
st.header(f"Date : {str_dt}" , divider="rainbow")

# Display site header
image_path = "../images/Flux_Dev_Create_a_image_of_for_a_sophisticated_Loan_Default_Ap_2.jpeg"
try:
    # Open and display the image
    img = Image.open(image_path)
    st.image(img, use_column_width =False, caption="Predicting Loan Defaults with AI/ML")  # Caption and resizing options
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

path_python_material = ".." # REPLACE WITH YOUR PATH

st.markdown("""
# Objective
## Predict if applicant will **Loan Default.**
- **Defaulted Loan:** A loan is considered in default when the borrower fails to make the required payments as agreed in the loan contract. 
    Default typically occurs after missing several payments (usually 90 to 180 days, depending on the type of loan and lender policies).
            
- **A loan is charged off** when the lender writes off the loan as a bad debt on their financial statements, recognizing it as a loss. 
    This typically happens after the loan has been in default for a significant period, often around 180 days.
---
## **The Loan Default Predictor:**
- Predicting if a applicants, if granted a loan will leads to a loan default or not.
---
            
- **Loan Default = False :** Applicant will repay back the loan -> *Lending Club Financial Gain !*
    - This will be provided with a percentage of how likely a applicant will have classed as Loan Default = False
    - Higher the percentage, lower the risk
            
---

- **Loan Default = True :** Applicant will NOT repay back the loan -> *Lending Club Financial Loss !*
    - This will be provided with a percentage of how likely a applicant will have classed as Loan Default = True  
    - Higher the percentage, higher the risk    

---  

""")

# Select Loan Default Model
st.header("Select Loan Default Predictor", divider=True)

# Load the pipeline model
for path, dir, files in os.walk(path_python_material):
    if 'models' in path:
        models = files

model = st.selectbox("Select Model", models, index=None,
    placeholder="Select a Machine Learning Classifier Model...",)

#if 'base_logistic_regression' in model:

if model:

    with open(f"{path_python_material}/models/{model}" , 'rb') as model_file:
        load_model = pickle.load(model_file)
    
    log_loan_amnt	=st.number_input('Loan Amount ( The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value. )', min_value=10)
    log_installment	=st.number_input('Installments', min_value=1)
    log_annual_inc	=st.number_input('Annual Income ' , min_value=10)
    log_fico_range_low	=st.number_input('FICO Range Low \n ( The upper boundary range the borrower’s FICO at loan origination belongs to. )')
    log_fico_range_high	=st.number_input('FICO Range High \n ( The lower boundary range the borrower’s FICO at loan origination belongs to. )')
    log_total_acc	=st.number_input('Total Acc ( The total number of credit lines currently in the borrowers credit file )')
    log_total_bc_limit	=st.number_input('Total Banck Card Limit ( Total bankcard high credit/credit limit )')
    log_int_rate_num	=st.number_input('Interest Rate %', min_value=1.0)

    dti	=st.number_input('dti ( A ratio calculated using the borrowers total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowers self-reported monthly income. )')

    term_months	=st.selectbox('term_ 36 or 60 months',
                                    ("36 months" , "60 months")
                                    )


    grade = st.selectbox('Grade (Lending Club assigned loan grade)',
                                    ('grade_A',
                                    'grade_B',
                                    'grade_C',
                                    'grade_D',
                                    'grade_E',
                                    'grade_F',
                                    'grade_G',)
                                    )

    emp_length_years = st.selectbox('Employment Length Years',
                                    ('emp_length_0 year',
                                    'emp_length_1 year',
                                    'emp_length_2 years',
                                    'emp_length_3 years',
                                    'emp_length_4 years',
                                    'emp_length_5 years',
                                    'emp_length_6 years',
                                    'emp_length_7 years',
                                    'emp_length_8 years',
                                    'emp_length_9 years',
                                    'emp_length_10+ years')
                                    )

    home_ownership = st.selectbox(   'Home Owernship Status',
                                ('home_ownership_ANY',
                                'home_ownership_MORTGAGE',
                                'home_ownership_NONE',
                                'home_ownership_OTHER',
                                'home_ownership_OWN',
                                'home_ownership_RENT'
                                )
                                )


    purpose = st.selectbox(   "Purpose for Loan",
                                (
                                'purpose_car',
                                'purpose_credit_card',
                                'purpose_debt_consolidation',
                                'purpose_home_improvement',
                                'purpose_house',
                                'purpose_major_purchase',
                                'purpose_medical',
                                'purpose_moving',
                                'purpose_other',
                                'purpose_renewable_energy',
                                'purpose_small_business',
                                'purpose_vacation',
                                'purpose_wedding'
                                )
                                )

    application_type  = st.selectbox(   "Application type (Joint or Individual)",
                                (
                                    "application_type_Individual",
                                    "application_type_Joint App"
                                )
                                )

    # Calculated fields
    log_debt_to_income_ratio	= np.log1p(log_loan_amnt/log_annual_inc) #st.number_input('log_debt_to_income_ratio')

    log_total_payment = 1.0

    if term_months == "36 months":
        log_total_payment	= np.log1p(log_installment * 36)

    if term_months == "60 months":
        log_total_payment	= np.log1p(log_installment * 60)

    data = {

            # numerical fields to transform to log-scale
            "log_loan_amnt"        :  np.log10(log_loan_amnt),  
            "log_installment"	   :  np.log1p(log_installment),	  
            "log_annual_inc"   :      np.log1p(log_annual_inc),  
            "log_fico_range_low":     np.log1p(log_fico_range_low),
            "log_fico_range_high" :   np.log1p(log_fico_range_high),
            "log_total_acc"	   :      np.log1p(log_total_acc),	  
            "log_total_bc_limit" :    np.log1p(log_total_bc_limit),
            "log_int_rate_num" : np.log10((log_int_rate_num)/100),



            # Calculated fields
            "log_total_payment" : log_total_payment,
            "log_total_interest" : log_total_payment - log_loan_amnt,
            "log_debt_to_income_ratio" : log_debt_to_income_ratio,

            # numerical not log-scaled
            "dti"	: dti,
            
            # Categorical fields
            "term__36_months": [1 if term_months == "36 months" else 0][0],
            "term__60_months": [1 if term_months == "60 months" else 0][0],

            "grade_A": [1 if grade=="grade_A" else 0][0],
            "grade_B": [1 if grade=="grade_B" else 0][0],
            "grade_C": [1 if grade=="grade_C" else 0][0],
            "grade_D": [1 if grade=="grade_D" else 0][0],
            "grade_E": [1 if grade=="grade_E" else 0][0],
            "grade_F": [1 if grade=="grade_F" else 0][0],
            "grade_G": [1 if grade=="grade_G" else 0][0],

            "emp_length_0_year" : [1 if emp_length_years  == "emp_length_0_years" else 0][0],
            "emp_length_10plus_years" :  [1 if emp_length_years == "emp_length_10+ years" else 0][0],
            "emp_length_1_year" : [1 if  emp_length_years == "emp_length_1_years" else 0][0],	
            "emp_length_2_years" : [1 if emp_length_years == "emp_length_2_years" else 0][0],
            "emp_length_3_years" : [1 if emp_length_years == "emp_length_3_years" else 0][0],
            "emp_length_4_years" : [1 if emp_length_years == "emp_length_4_years" else 0][0],
            "emp_length_5_years" : [1 if emp_length_years == "emp_length_5_years" else 0][0],
            "emp_length_6_years" : [1 if emp_length_years == "emp_length_6_years" else 0][0],
            "emp_length_7_years" : [1 if emp_length_years == "emp_length_7_years" else 0][0],
            "emp_length_8_years" : [1 if emp_length_years == "emp_length_8_years" else 0][0],
            "emp_length_9_years" : [1 if emp_length_years == "emp_length_1_years" else 0][0],


            "home_ownership_ANY" 	   : [1 if home_ownership == "home_ownership_ANY"  else 0][0],
            "home_ownership_MORTGAGE":[1 if home_ownership == "home_ownership_MORTGAGE" else 0][0],
            "home_ownership_NONE" 	:[1 if home_ownership == "home_ownership_NONE" else 0][0],
            "home_ownership_OTHER" 	:[1 if home_ownership == "home_ownership_OTHER" else 0][0],
            "home_ownership_OWN"	   :[1 if home_ownership == "home_ownership_OWN" else 0][0],
            "home_ownership_RENT" :[1 if home_ownership == "home_ownership_RENT" else 0][0],

            "purpose_car" : [1 if purpose == "purpose_car" else 0][0],
            "purpose_credit_card": [1 if purpose == "purpose_credit_card" else 0][0],
            "purpose_debt_consolidation": [1 if purpose == "purpose_debt_consolidation" else 0][0],
            "purpose_home_improvement": [1 if purpose == "purpose_home_improvement" else 0][0],
            "purpose_house": [1 if purpose == "purpose_house" else 0][0],
            "purpose_major_purchase":[1 if purpose == "purpose_major_purchase" else 0][0],
            "purpose_medical": [1 if purpose == "purpose_medical" else 0][0],
            "purpose_moving": [1 if purpose == "purpose_moving" else 0][0],
            "purpose_other": [1 if purpose == "purpose_other" else 0][0],
            "purpose_renewable_energy": [1 if purpose == "purpose_renewable_energy" else 0][0],
            "purpose_small_business": [1 if purpose == "purpose_small_business" else 0][0],
            "purpose_vacation": [1 if purpose == "purpose_vacation" else 0][0],
            "purpose_wedding": [1 if purpose == "purpose_wedding" else 0][0],

            "application_type_Individual" :[1 if application_type == "application_type_Individual" else 0][0],
            "application_type_Joint_App" : [1 if application_type == "application_type_Joint_App" else 0][0],

        }

    df_data = pd.DataFrame(data, index=[0])

    if 'base_logistic_regression' in model:
        print(f"Model LogisticRegression = {load_model.predict(df_data)}")
        
        proba_0 = round(load_model.predict_proba(df_data)[:,0][0] , 2) * 100
        proba_1 = round(load_model.predict_proba(df_data)[:,1][0] , 2) * 100
        st.header(f"Model LogisticRegression ... Loan Default Prediction = {load_model.predict(df_data)[0]}", divider=True)
        st.header(f"Probabilty of Loan Default = False : {proba_0:.2f} %", divider=False)
        st.header(f"Probabilty of Loan Default = True : {proba_1:.2f} %", divider=True)
        
        # Coefficients for logistic regression
        importance = load_model.coef_
        feature_names = load_model.feature_names_in_
        
        st.header(f"Feature Importance Chart", divider=True)
        st.markdown("""
        #### Showing which selection/choices have the biggest impact on the Final prediction
            """)
        df_features = pd.DataFrame(np.abs(importance), columns=feature_names)
        st.bar_chart(df_features, stack=False)

    elif 'best_xgb_rfe' in model:
        print(f"Model XGBoost = {load_model.predict(df_data[load_model.feature_names_in_]).astype(bool)}")
        
        proba_0 = round(load_model.predict_proba(df_data[load_model.feature_names_in_])[:,0][0], 3)*100.0
        #print(proba_0)
        proba_1 = round(load_model.predict_proba(df_data[load_model.feature_names_in_])[:,1][0], 2)*100.0
        #print(proba_1)
        
        st.header(f"Model XGBoost ... Loan Default Prediction = {load_model.predict(df_data[load_model.feature_names_in_]).astype(bool)[0]}", divider=True)
        st.header(f"Probabilty of Loan Default = False : {proba_0:.2f} %", divider=False)
        st.header(f"Probabilty of Loan Default = True : {proba_1:.2f} %", divider=True)

        # Coefficients for logistic regression
        importance = [load_model.feature_importances_]
        feature_names = load_model.feature_names_in_
        
        st.header(f"Feature Importance Chart", divider=True)
        st.markdown("""
        #### Showing which selection/choices have the biggest impact on the Final prediction
            """)
        df_features = pd.DataFrame(importance, columns=feature_names, index=[0])
        st.bar_chart(df_features, stack=False)
    
else:
    st.header("Select a model to start predicting loan defaults", divider=True)
