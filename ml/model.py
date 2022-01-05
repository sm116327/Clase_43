import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

df = pd.read_csv(f"{os.getcwd()}/data/iris.csv")

X = df.drop("variety", axis=1).values.copy()
y = df.variety.copy()

clf = RandomForestClassifier(max_depth=2)
print("entrenando modelo")
clf.fit(X, y)

pickle.dump(clf, open(f"{os.getcwd()}/ml/model.pkl", 'wb'))