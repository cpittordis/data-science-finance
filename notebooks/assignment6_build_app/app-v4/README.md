# 0. App version log


| Version | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1      | Initial release. Basic functionality implemented.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| v2      | Upgrading Streamlit application by adding interactive visualizations.<br>Including a feature importance graph to help users understand which features are most influential in predicting fraud.<br>Implementing real-time confidence intervals to display the model's prediction uncertainty. <br>Integrating SHAP (SHapley Additive exPlanations) into the application to provide explanations for each prediction. <br>Displaying SHAP values for individual transactions so that users can see which features contributed to the prediction. |
| v3      | Modified FastAPI application to handle real-time data streams.<br>Implemented a feature that allows the model to continuously score incoming transactions from a simulated real-time data source (csv in target)                                                                                                                                                                                                                                                                                                                                 |
| v4  <!--(Charalambos Pittordis Edits)-->    | Uploading .CSV file(s), a new feature (transaction_amount_to_balance_ratio) is created automatically, and predictions can be made on the entire uplaoded file in one go (file with predictions downloadable), including their confidence counterpart (i.e., probability of scoring 0 or 1 for each record). This follows with a interactive scatter plot where the themes, colours, features, hues can be chosen. The additional sections utilises SHAPASH, allowing for interactive feature importances and presenting the model peformance and numerical contributions to each feature for each records (file with feature contributions downloadable). The last section contains <Manual scoring of individual transactions> the flexibilty of a user to place in bespoke values for transaction_amount, customer_age, customer_balance returning a prediction value with visuals. |

# 1. Running the app

To start the app, run the following commands **in 2 different terminal windows**

Within both terminals, ensure you are within the app-v4 parent directory, accessing the 'src' sub-directory: 
- e.g, "../path/to/app-v4/src/" 

### Within both terminals run:
terminal window 1: `uvicorn dsif11app-fraud-backend:app --reload`

terminal window 2: `streamlit run dsif11app-fraud-streamlit.py`

This will load up the app.

**NOTE**: All tables displayed, if you hover the cursor to the top-right of the table, there are 3 functions, and one of them gives you the availability to download the data. 

---

## 1.1 Uploading .CSV

Once the application is launched, the frst header will request for a file to be uploaded.
Ensure the file is a .csv, and contains the fields within the format `transaction_amount`, `customer_age`, `customer_balance`; as these are the fields that are used within the Machine Learning model t make predictions. in addition, once the file is uploaded, a **New Feature: Transaction Amount to Balance Ratio** `transaction_amount_to_balance_ratio` will be produced.


## 1.2 Check-box to make predictions.

This section gives the user the flexibility to produce the predictions on the entire uploaded file, with its probability counterparts.

## 1.3 Interactive Scatter Plot.

This allows the user to select which fields they will like to plot, and be able to customise the theme, hues, ...etc, also being able to download the plot itself by hovering the cursor to the top-right of the plot.

## 1.4 Interactive Plots of Feature Importances & Contributions.

This section utilises the python package [SHAPASH](https://shapash.readthedocs.io/en/latest/index.html), where it produces interactive plots of feature importances & contributions with the flexibility of selecting a feature to see how it performs across all records. This help users understand which features are most influential in predicting fraud. In addition, it produces a feature importance & contributions dataset which is downloadable, and a confusion matrix scatter plot to show those predicted correctly or not. This section allows the user to select the number of records to visualise (default is 10 records, and any number larger than the size of the dataset will default to the full dataset). All plots are downloadable.

## 1.5 Manual Scoring of Individual Transactions.

This section allows the user to input bespoke values to whether the outcome is most likely fraud or not. This includes to help them understand which features importance graph to help users understand which features are most influential in predicting fraud.
Implementing real-time confidence intervals to display the model's prediction uncertainty.
Integrating SHAP (SHapley Additive exPlanations) into the application to provide explanations for each prediction.
Displaying SHAP values for individual transactions so that users can see which features contributed to the prediction


<!-- 
--- 
# 2. Assignment Instructions

## **Exercise 2.1**: Adding a File Upload Section and Saving Predictions in CSV (**5 points**)

In this exercise, you will extend your Streamlit app by allowing users to upload a file with new transactions, get the fraud predictions back, and save the results as a CSV file. Follow the instructions below to complete the task.

### **Step-by-Step Instructions:**

a. **File Upload Section**:

* Add a section where users can upload a CSV file containing new transactions.
* The file should include columns such as `transaction_amount`, `transaction_time`, `customer_age`, `customer_balance`.
  b. **Process the File**:
* Once the file is uploaded, read it into a pandas DataFrame.
  c. **Run Predictions**:
* After the file is uploaded, run predictions using your pre-trained model for all transactions in the file. You can do this by either calling the `predict/` HTTP method we built previously (defined in  `dsif11app-fraud.py`) for each individual transaction and process it one at the time, or by creating an additional HTTP method that does the batch scoring (similar to what we implemented in the `predict_automation/` method).
  d. **Save Predictions**:
* Allow users to download the results (with fraud predictions) as a CSV file.
* Include an option to choose the location where the file will be saved.

## **Exercise 2.2**:   Adding Visuals to Streamlit App (5 points)

In this exercise, you will enhance your Streamlit app by incorporating an interactive scatter plot feature. This will allow users to select which numerical columns from the dataset to display against each other for deeper insights. You will also create an additional feature, **transaction amount to balance ratio**, which can be included as a selectable option for visualization.

### **Step-by-Step Instructions:**

a.  **New Feature: Transaction Amount to Balance Ratio**:
-   Create a new feature: **transaction amount to balance ratio**.
-   Include this feature as one of the selectable options for the scatter plot.
b.  **Interactive Scatter Plot**:
-   Add an interactive scatter plot to the app that lets users choose which numerical columns to plot on the x-axis and y-axis from the available dataset columns.

This exercise will give users more flexibility in exploring relationships between different variables and uncovering patterns in transaction data.

# 3 Additional exercises (optional)

If you would like to move further on the 'app building' journey, feel free to use the exercises below as an opportunity to practice something new.

**NOTE:** This will likely require additional reading online and some of the below concepts are definitely more geared towards medium-advanced level Pythonists!

## 3.1 Cloud Deployment and Monitoring

- Deploy your FastAPI application to a cloud platform (e.g., AWS, Azure, Heroku). Set up basic monitoring to log and analyze prediction results over time. Provide a brief report on the application's performance post-deployment.

## 3.2 Bias Analysis and Ethical Implications

- Analyze your model for potential bias, especially regarding customer demographics (i.e. age). Discuss any ethical implications of using your model in a real-world setting and propose strategies to mitigate identified biases if any (e.g. rebalancing the training dataset or adjusting model thresholds).

## 3.3 Security Measures in FastAPI

- Implement at least two security measures in your FastAPI application to protect it from common vulnerabilities. These could include input validation, rate limiting, or basic authentication.

  > e.g.:
  >

  - Add input validation to ensure that only valid data is processed by the API.
  - Implement rate limiting to prevent abuse by limiting the number of requests from a single IP address.

## 3.4 Comprehensive Documentation and Testing

- Write detailed documentation for your FastAPI application, including instructions on deployment, usage, and extending the model. Implement unit and integration tests to ensure your API endpoints and model predictions are functioning correctly.
- **Example:** Create a README file that explains how to set up and deploy the FastAPI app, including example API calls. Write unit tests using a testing framework (e.g., pytest) to validate API behavior.

## 3.5 Building fraud model using more realistic data

In this exercise, you will implement a fraud detection model using autoencoders, leveraging the **Credit Card Fraud Detection** dataset from Kaggle. This dataset contains transactions made by European cardholders in 2013, with a small percentage flagged as fraudulent. Autoencoders are particularly well-suited for anomaly detection, which is essential for identifying fraudulent transactions.

### Key information:

- **Example notebooks** for autoencoder model in fraud, including useful explanation on autoencoders, and link to Kaggle which includes both notebooks and dataset: [here](https://towardsdatascience.com/detection-of-credit-card-fraud-with-an-autoencoder-9275854efd48).

### **Step-by-Step Instructions:**

a. **Dataset Selection**:

* Use the **Credit Card Fraud Detection** dataset from Kaggle for training your autoencoder. It includes the `Class` (fraud flag) column.
  b. **Data Preprocessing:**
* Normalize numerical columns such as Amount and Time to ensure all inputs are on a similar scale.
  c. **Separate Fraudulent and Non-Fraudulent Transactions:**
* Autoencoders are trained to reconstruct non-fraudulent transactions, so separate out the Class = 0 (non-fraud) transactions.
  d. **Build, train and evaluate the autoencoder model:**
* Using a deep learning framework like Keras, create an autoencoder model with an encoder (for compression) and a decoder (for reconstruction). Save the model in the relevant folder.
  e. **File Upload Section**:
* Add a section in the Streamlit app where users can upload a CSV file containing new transactions.
* Ensure the file includes relevant columns as per schema that is expected.
  f. **Process the File**:
* Once the file is uploaded, read it into a pandas DataFrame for processing.
  g. **Run Predictions**:
* Run predictions using your trained autoencoder model for each transaction in the uploaded file.
* You can use the existing HTTP method (e.g., `predict/` in `dsif11app-fraud.py`) to handle individual transactions or implement a batch prediction method for more efficient processing.
  h. **Save Predictions**:
* Allow users to download the fraud prediction results as a CSV file.
* Include an option for users to choose the location where the file will be saved locally.
-->