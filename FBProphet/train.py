import config
from dataset import df, df2
from model_fbprophet import model_fb

import joblib
import json
from fbprophet.serialize import model_to_json

def run():

    # Define Train Dataset
    X_train = df2[df2['ds'] <= config.END_DATE]
    # X_train = df2[['ds', 'y']]
    # X_train.set_index('ds', inplace = True)
    # print(X_train.head())

    # Load Model
    model = model_fb

    # Train the model
    print("\n---------- Model Training initiated. ----------\n")

    model.fit(X_train)

    print("\n---------- Model Training complete. ----------\n")

    # Save the model
    with open(config.MODEL_PATH, 'w') as fout:
        json.dump(model_to_json(model), fout)
    print("\n---------- Model saved. ----------\n")

if __name__ == "__main__":
    run()