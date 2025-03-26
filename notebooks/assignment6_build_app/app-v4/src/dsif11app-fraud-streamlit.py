

api_url = "http://127.0.0.1:8000"

import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import pickle
import sys
import pandas as pd
import datetime
from shapash import SmartExplainer
import json

st.set_page_config(layout="wide")
css = '''
<style>
        .stMainBlockContainer {
            max-width:90rem;
        }
    </style>
'''
st.markdown(css, unsafe_allow_html=True)

# set datetime string

str_dt = datetime.datetime.now().strftime('%Y-%m-%d') ## String on DateTime in format YYYYMMDD

# Create figure and add one scatter trace
fig = go.Figure()

# Title
st.title("Fraud Detection App")
st.header(f"Date : {str_dt}" , divider="rainbow")

# Display site header
#image = Image.open("../images/dsif header.jpeg")

image_path = "../images/dsif header 2.jpeg"
try:
    # Open and display the image
    img = Image.open(image_path)
    st.image(img, use_container_width =False)  # Caption and resizing options
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

path_python_material = ".." # REPLACE WITH YOUR PATH
model_id = "lr1"

# Load the pipeline model
with open(f"{path_python_material}/models/{model_id}-pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Specifying columns for predictions
col_for_predcitions = ['transaction_amount', 'customer_age', 'customer_balance']

@st.cache_data
def make_pred(df, col_for_predcitions):

    
    # Legacy -> using code within streamlit
    #predictions = model.predict(df[col_for_predcitions])
    #predictions_proba_0 = model.predict_proba(df[col_for_predcitions])[:,0]
    #predictions_proba_1 = model.predict_proba(df[col_for_predcitions])[:,1]

    # New -> using the api call

    # converting dataframe to dictionary
    data = df[col_for_predcitions].to_dict(orient="records")

    # calling api to make predictions on dataframe
    response = requests.post(f"{api_url}/predict_df", json= {"data" : data})

    # gathering the predicted column we want for our dataset and app
    predictions = response.json()['predictions']
    predictions_proba_0 = response.json()['predictions_proba_0']
    predictions_proba_1 = response.json()['predictions_proba_1']

    # creating new dataframe columns containing predicted values from api call
    df['prediction'] = predictions
    df['prediction_prob_0'] = predictions_proba_0
    df['prediction_prob_1'] = predictions_proba_1

    return df

def data_to_view(uploaded_file):

    df_file = pd.read_csv(uploaded_file)
    df = df_file

    # New feature
    st.write("Creating new feature = transaction_amount_to_balance_ratio")
    df['transaction_amount_to_balance_ratio'] = round(df['transaction_amount'] / df['customer_balance'] , 3)

    if make_pred_check:
        with st.spinner("In Predictions in progress ..."):

            df_pred = make_pred(df, col_for_predcitions)
            st.write("Data with Predictions:") 
            st.write("See Columns ['prediction', 'prediction_prob_0', 'prediction_prob_1']")      
            df = df_pred
            st.dataframe(df) 
        st.success("Predictions made ..!!   --> See Columns ['prediction', 'prediction_prob_0', 'prediction_prob_1']")
    else:
        st.write("Uploaded Data:")
        df
    
    return df


# File upload
st.header("File upload", divider=True)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

st.subheader("Check if you want to make Predictions on Dataset")
make_pred_check = st.checkbox("Make prediction")

if uploaded_file:

    df = data_to_view(uploaded_file)

else:
    df = pd.DataFrame()

# Scatter plot
st.header("Scatter Plot for Dataset", divider=True)

# Theme selection
theme = st.selectbox("Select Plot Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"])

# Column selection
x_col = st.selectbox("Select X-axis column", df.columns)
y_col = st.selectbox("Select Y-axis column", df.columns)
hue_col = st.selectbox("Select hue column (optional)", [None] + list(df.columns)) #Allow no hue
hover_col = st.selectbox("Select hover data column", df.columns)
size_col = st.selectbox("Select size column (optional)", [None] + list(df.columns))

# Visual of the scatter plot
if x_col and y_col and hover_col:
    try:
        # Set the Plotly theme
        fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, hover_data=[hover_col], size=size_col, template=theme)
        st.plotly_chart(fig, theme=None)
    except Exception as e:
        st.error(f"Error plotting: {e}")


# Interactive plots and feature importance
st.header("Interactive Plots of Feature Importances & Contributions", divider=True)
st.subheader("Check if you want to Make Interactive Plots with Feature Importance & Dataset")
interactive_features_plots_and_data = st.checkbox("Make Interactive Plots with Feature Importance & Dataset")

if interactive_features_plots_and_data:
    
    response_dict = {0: 'No Fraud', 1:' Fraud Detected'}

    # Calling SmarExplainer to perform on `model`
    xpl = SmartExplainer(model=model , label_dict=response_dict)

    data_size = st.number_input("Select a number of records for Interacive plotting and Feature Importance ..... NOTE: these will be selected randomly.")
    
    if data_size >= len(df):
        df_sample = df
    elif data_size < 10:
        df_sample = df.sample(10)
    else:
        df_sample = df.sample(int(data_size))

    
    with st.spinner("In progress"):

        # Compiling SmartExplainer on the testing data
        xpl.compile(x=df_sample[col_for_predcitions], 
                    y_target=df_sample['is_fraud'].astype(int), # Optional: allows to display True Values vs Predicted Values
                )
    st.success("Done!")
        
    df_xpl = xpl.to_pandas().reset_index(drop=True)

    st.subheader("Data with Feature Importance", divider=True)
    st.dataframe(df_xpl)

    # Feature Importance
    st.subheader("Feature Importance", divider=True)
    plot_feat = xpl.plot.features_importance()
    st.plotly_chart(plot_feat)

    st.subheader("Feature Contributions", divider=True)
    feature_sel = st.selectbox("Select feature", col_for_predcitions)
    plot_feat_cont = xpl.plot.contribution_plot(feature_sel)
    st.plotly_chart(plot_feat_cont)

    # Confusion Matrix Scatter
    st.subheader(f"Data Sample of {len(df_sample)} records ", divider=True)
    st.dataframe(df_sample)

    scat_lot_pred_0 = xpl.plot.scatter_plot_prediction(label=response_dict[0])
    st.plotly_chart(scat_lot_pred_0)

    scat_lot_pred_1 = xpl.plot.scatter_plot_prediction(label=response_dict[1])
    st.plotly_chart(scat_lot_pred_1)

# Input bespoke values
st.header("Manual Scoring of Individual Transactions", divider=True)
st.subheader("Check if you want to Make predictions on input bespoke values")
bespoke_predictions = st.checkbox("Make predictions on input bespoke values")

if bespoke_predictions:

    transaction_amount = st.number_input("Transaction Amount")
    customer_age = st.number_input("Customer Age")
    customer_balance = st.number_input("Customer Balance")

    data = {
            "transaction_amount": transaction_amount,
            "customer_age": customer_age,
            "customer_balance": customer_balance
        }

    if st.button("Show Feature Importance"):
        
        response = requests.get(f"{api_url}/feature-importance")
        feature_importance = response.json().get('feature_importance', {})

        features = list(feature_importance.keys())
        importance = list(feature_importance.values())

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        st.pyplot(fig)

    if st.button("Predict and show prediction confidence"):
        # Make the API call

        response = requests.post(f"{api_url}/predict/",
                                json=data)
        result = response.json()
        confidence = result['confidence']

        if result['fraud_prediction'] == 0:
            st.write("Prediction: Not fraudulent")
        else:
            st.write("Prediction: Fraudulent")

        # Confidence Interval Visualization
        labels = ['Not Fraudulent', 'Fraudulent']
        fig, ax = plt.subplots()
        ax.bar(labels, confidence, color=['green', 'red'])
        ax.set_ylabel('Confidence')
        ax.set_title('Prediction Confidence')
        st.pyplot(fig)

    if st.button("Predict and show SHAP values"):
        response = requests.post(f"{api_url}/predict/",
                                json=data)
        result = response.json()

        if result['fraud_prediction'] == 0:
            st.write("Prediction: Not fraudulent")
        else:
            st.write("Prediction: Fraudulent")

        ######### SHAP #########
        # Extract SHAP values and feature names
        shap_values = np.array(result['shap_values'])
        features = result['features']

        # Display SHAP values
        st.subheader("SHAP Values Explanation")

        # Bar plot for SHAP values
        fig, ax = plt.subplots()
        ax.barh(features, shap_values[0])
        ax.set_xlabel('SHAP Value (Impact on Model Output)')
        st.pyplot(fig)
