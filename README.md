# Employee Attrition Prediction and Analysis Using a Light Gradient Boosting Machine
Performing employee attrition prediction and model analysis using data from exit interviews, performance reviews, and employee records. Also saves trained model and used on a test file to demonstrate usage on future test data.

Data was sourced from a US-based company whose name was withheld for security reasons and provided by Marika Stewart in https://www.kaggle.com/datasets/marikastewart/employee-turnover.

Data exploration was undertaken to determine distribution and spread. Univariate outliers existed and suggested that multivariate outliers may also be present, at least within the continuous features. Feature selection was undertaken through the Boruta method, and the chosen features were all continuous. Multivariate outliers were identified through analysis of Mahalanobis distance and removed. 

Multiple models were compared, and best performing model was a Light Gradient Boosted Machine (Light GBM). Through analyzing feature importance and SHAP values from model predictions, most important features in contributing to employee churn were average hours rendered per week, employee performance rating, and  employee satisfaction.  

Model test AUC is at 0.922, and test accuracy is at 84.22%, with recall at 81.88%. 
