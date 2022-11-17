from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
from plotKNN import plotKNN

## load and split
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split( iris_data.data, iris_data.target, test_size = 0.3, random_state = 100 )

## create model
sk_knn = KNeighborsClassifier(n_neighbors=5)

## train
sk_knn.fit(X_train, y_train)

## predict
y_hat = sk_knn.predict(X_test)

## evaluate
print("Accuracy:",metrics.accuracy_score(y_test, y_hat))

## plotting
plotKNN(iris_data, X_test, y_test, y_hat)