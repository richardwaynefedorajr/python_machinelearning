import time
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns

## initializations
seed = random.randint(1,10) # or set to constant2..0020000
n_clusters = 7
n_centers = 7

## create dataset
X, y = make_blobs(n_samples=1500, random_state=seed, centers=n_centers)

## iterate over range for number of clusters to fit
for i in range(n_clusters):
    clear_output(wait=True)
    
    ## train
    sk_kmeans = KMeans(n_clusters=i+1, random_state=seed, init='k-means++', n_init=1, max_iter=10).fit(X)
    
    ## predict
    y_hat = sk_kmeans.predict(X)
    centroids = sk_kmeans.cluster_centers_
    distances = 1 - preprocessing.MinMaxScaler(feature_range=(1e-5,1-1e-5)).fit_transform(np.linalg.norm(X - centroids[y_hat,:], axis=1).reshape(-1,1))
    
    ## plotting
    df = pd.DataFrame(data=np.c_[X, y_hat], columns=['X_x','X_y','y_hat'])
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(data=df, x='X_x', y='X_y', hue='y_hat', palette='viridis', alpha=distances, legend='full')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.scatter(centroids[:, 0], centroids[:, 1], edgecolor='black', s=35, c='r', marker='D')
    plt.title("K-Means Clustering: Iteration {}".format(i+1))
    plt.show()
    time.sleep(1)