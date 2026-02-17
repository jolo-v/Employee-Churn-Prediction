# Employee Attrition Prediction and Analysis Using a Light Gradient Boosting Machine
Performing employee attrition prediction and model analysis using data from exit interviews, performance reviews, and employee records. Also saves trained model and used on a test file to demonstrate usage on future test data.

Data was sourced from a US-based company whose name was withheld for security reasons and provided by Marika Stewart in https://www.kaggle.com/datasets/marikastewart/employee-turnover.

Data exploration was undertaken to determine distribution and spread.  Feature selection was undertaken through the Boruta method, and the chosen features were all continuous. 

Multiple models were compared, and best performing model was a Light Gradient Boosted Machine (Light GBM). Through analyzing feature importance and SHAP values from model predictions, most important features in contributing to employee churn were average hours rendered per week, employee performance rating, and  employee satisfaction.  

Model test AUC is at 0.922, and test accuracy is at 84.22%, with recall at 81.88%. 

The model could be hosted in an endpoint using the Dockerfile and predict script. A script loading new test data can then invoke endpoint through an API call for new predictions.
