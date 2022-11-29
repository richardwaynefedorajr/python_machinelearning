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
n_clusters = 8
n_centers = 4

## plotting initializations
n_subplot_cols = 4
n_subplot_rows = 2
fig, axs = plt.subplots(ncols=n_subplot_cols, nrows=n_subplot_rows, figsize=(40,20))

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
    ax_current = axs[np.unravel_index(i, (n_subplot_rows, n_subplot_cols))]
    sns.scatterplot(data=df, x='X_x', y='X_y', hue='y_hat', palette='viridis', alpha=distances, legend='full', 
                         ax=ax_current)
    # sns.move_legend(ax_current, "upper left", bbox_to_anchor=(1, 1))
    ax_current.scatter(centroids[:, 0], centroids[:, 1], edgecolor='black', s=75, c='r', marker='D')
    ax_current.set_title("# of clusters = {}".format(i+1), fontsize=16)
    # plt.show()
    time.sleep(1)
    
fig.suptitle('K-means clustering with {} data blobs and iterating to {} cluster means'.format(n_centers, n_clusters), fontsize=24)
fig.savefig('k-means.png', bbox_inches='tight')