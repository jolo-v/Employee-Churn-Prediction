# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 14:29:13 2025

@author: jolo_
"""

import requests
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

#Define endpoint
url = 'http://127.0.0.1:9696'

#Read csv file
data = pd.read_csv('employee_churn_test.csv')
data = data.iloc[:,1:]

#Convert to json
json_payload = data.to_dict(orient='records')

#Send json to endpoint
response = requests.post(url, json=json_payload)
data_response = response.json()

#Process response
final_data = pd.DataFrame(data_response)
final_data = final_data[['review','tenure','satisfaction','avg_hrs_month','left','Preds']]

#Compute netrics
accuracy_score(y_true=final_data['left'],y_pred=final_data['Preds'])
roc_auc_score(y_true=final_data['left'],y_score=final_data['Preds'])
f1_score(y_true=final_data['left'],y_pred=final_data['Preds'])
recall_score(y_true=final_data['left'],y_pred=final_data['Preds'])
