import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

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

## create dataframes for visualization
iris_df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']], columns=iris_data['feature_names'] + ['target'])

## histograms
fig, axs = plt.subplots(2, 2, figsize=(7, 7))
sns.histplot(data=iris_df, x=iris_data.feature_names[0], bins=50, color="chartreuse", ax=axs[0, 0])
sns.histplot(data=iris_df, x=iris_data.feature_names[1], bins=50, color="lawngreen", ax=axs[0, 1])
sns.histplot(data=iris_df, x=iris_data.feature_names[2], bins=50, color="green", ax=axs[1, 0])
sns.histplot(data=iris_df, x=iris_data.feature_names[3], bins=50, color="lime", ax=axs[1, 1])
plt.savefig('knn_histograms.png')
plt.show()

## correlation matrix
ax = sns.heatmap(iris_df[iris_data.feature_names].corr(), cmap="Greens", annot=True, square=True)
ax.set_ylim(0, len(iris_df[iris_data.feature_names].corr()))
plt.savefig('knn_corr_matrix.png')
plt.show()

## pairplots
iris_df['target'][iris_df["target"] == 0.0] = iris_data.target_names[0]
iris_df['target'][iris_df["target"] == 1.0] = iris_data.target_names[1]
iris_df['target'][iris_df["target"] == 2.0] = iris_data.target_names[2]
sns.pairplot(iris_df, hue="target")
plt.savefig('knn_pairplots.png')
plt.show()