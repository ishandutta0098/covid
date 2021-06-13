import config
from dataset import df, df2

import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score

import json
from fbprophet.serialize import model_from_json

# Load Model
with open(config.MODEL_PATH, 'r') as fin:
    model = model_from_json(json.load(fin))
print("\n---------- Model loaded for inference. ----------\n")

# print(df2.head())
# # Define test data
X_test = df2[df2['ds'] > config.END_DATE]
# # X_test = df2[df2['ds'] < '2021-04-02']
# X_test = df2[df2['ds'] >= '2021-05-16']
# # X_test = df2[['ds', 'y']]
# X_test.set_index('ds', inplace = True)
# print(X_test.isnull().sum())

# Obtain Predictions
print("\n---------- Model is calculating predictions. ----------\n")
# preds = model.predict(X_test[['ds']])
preds = model.predict(X_test)
print("\n---------- Model Predictions calculation complete. ----------\n")


# Evaluate Model using Root Mean Squared Error
# print("actual - ", np.exp(X_test['y']))
# print("predicted - ",np.exp(preds['yhat']))
rmse = np.sqrt(mean_squared_error(np.exp(X_test['y']), np.exp(preds['yhat'])))
accuracy = accuracy_score(X_test['y'], preds['yhat'])
print(f"RMSE Score: {rmse}")
print("accuracy", accuracy)