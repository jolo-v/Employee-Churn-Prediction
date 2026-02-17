# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 11:25:28 2025

@author: jolo_
"""

import pickle
import lightgbm as lgbm
import pandas as pd
from flask import Flask, request, jsonify


#Load model
with open('employee_churn_model.pkl', 'rb') as gbm:
    model = pickle.load(gbm)

#Load column transformer
with open('column_transformer.pkl', 'rb') as col_trans:
    col_transformer = pickle.load(col_trans)

#Process csv input
def process_input(input_):
    #data = pd.read_json(input_)
    data = pd.DataFrame(input_)
    test = col_transformer.transform(data)
    preprocessed_test = pd.DataFrame(test,
                                columns=['IT', 'admin', 'engineering', 'finance', 'logistics', 'marketing', 'operations', 'retail', 'sales', 'support', 'review', 'projects', 'tenure', 'satisfaction', 'avg_hrs_month', 'promoted', 'salary', 'bonus', 'left'])
    test_filtered = preprocessed_test[['review','tenure','satisfaction','avg_hrs_month','left']]
    recode = {'yes':1,'no':0}
    test_filtered['left'] = test_filtered['left'].map(recode)
    return test_filtered
  
#Predict output of data  
def predict(data):
    pred = model.predict(data.iloc[:,:-1].values)
    preds = pd.DataFrame(pred,columns=['Preds'])
    return preds

#Process output
def process_output(input_,pred):
    output_ = pd.concat([input_,pred],axis=1)
    return output_

#Define flask app
app = Flask('Employee_Churn')

#Define main function
@app.route('/', methods=['POST'])
def predict_endpoint():
    input_ = request.get_json()
    newdata = process_input(input_)
    preds = predict(newdata)
    output = process_output(newdata,preds)
    output_send = output.to_dict(orient='records')
    return output_send

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='9696')