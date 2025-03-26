from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def model_evaluation_report_classification(X_test, y_test, y_pred, y_prob):
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) #classifier.predict_proba(X_test)[:,1])

    # Display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall (Sensitivity): {recall}')
    print(f'F1-Score: {f1}')
    print(f'ROC-AUC: {roc_auc}')
    print(f'Confusion Matrix:\n{cm}')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def model_evaluation_report_regression(y_target, y_pred):
    
    # Calculating Metrics
    mae = mean_absolute_error(y_target, y_pred)
    mse = mean_squared_error(y_target, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_target, y_pred)

    print("\n ... Evaluation Metrics for Regression ... ")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
