# Simple decision tree

import scipy as scp
import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier

# Test (coursera)

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

clf = DecisionTreeClassifier()
clf.fit(X, y)

importances = clf.feature_importances_
print(importances)

# Titanic (coursera)

data = pd.read_csv("titanic.csv", sep = ",")
print(data.describe()) # check if we've loaded right data

data_notna = pd.DataFrame.dropna(data)
print(data_notna.describe())

X = data_notna[["Pclass", "Fare", "Age", "Sex"]]
X.replace("male", 0, True, None, False)
X.replace("female", 1, True, None, False)
print(X.describe)

y = data_notna["Survived"]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)