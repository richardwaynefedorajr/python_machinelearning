from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from plotKNN import plotKNN
from tuneHyperParameters import tuneHyperParameters

## load and split
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split( iris_data.data, iris_data.target, test_size = 0.3, random_state = 100 )

## tune hyperparameters
optimal_k = tuneHyperParameters(X_train, y_train)

## create model
sk_knn = KNeighborsClassifier(n_neighbors=optimal_k)

## train
sk_knn.fit(X_train, y_train)

## predict
y_hat = sk_knn.predict(X_test)

## plot and evaluate
plotKNN(iris_data, X_test, y_test, y_hat, optimal_k)