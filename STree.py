# Simple decision tree

import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


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
print(X[:5])

y = data_notna["Survived"]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

print("Do they survived?")
n = 1
print(clf.predict(X[:5]))

print(clf.feature_importances_)
print(cross_val_score(clf, X, y, cv=10))

plt.figure()
plt.title("tree")
plt.xlabel("Training examples")
plt.ylabel("Score")
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=241)
train_sizes=np.linspace(.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=4, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")

plt.show()