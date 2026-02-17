#Use official Python 3.12 image as base image
FROM python:3.12-slim

#Copy list of dependencies
COPY requirements.txt ./

# Install Python libraries
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

#Set working directory
WORKDIR /app

#Copy python script and model
COPY ["Churn_Predict_Script.py","employee_churn_model.pkl","column_transformer.pkl", "./"]

#Expose port
EXPOSE 9696 

#Run script
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "Churn_Predict_Script:app"]