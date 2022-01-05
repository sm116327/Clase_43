import pickle
import sys
import pandas as pd
import os


def predict(data):
    model = pickle.load(open(f"{os.getcwd()}/ml/model.pkl", 'rb'))
    preds = model.predict(data)
    preds = pd.DataFrame(preds)
    preds.to_csv(f"{os.getcwd()}/data/predictions.csv")


if __name__ == '__main__':
    data_dir = sys.argv[1]
    data = pd.read_csv(f"{os.getcwd()}{data_dir}")
    data = data[["sepal.length", "sepal.width",  "petal.length",  "petal.width"]].values
    predict(data)