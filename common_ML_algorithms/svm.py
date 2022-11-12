import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager

## import data
wine = datasets.load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

## train
sk_svm = svm.SVC(kernel='linear').fit(X_train, y_train)

## predict
y_hat = sk_svm.predict(X_test)

## accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_hat))
print("Precision:",metrics.precision_score(y_test, y_hat, average='weighted'))
print("Recall:",metrics.recall_score(y_test, y_hat, average='weighted'))

## add data to plot
cmap = cm.get_cmap('viridis')
fig = plt.figure()
ax = plt.axes()
ax.scatter(X_test[:,0], X_test[:,1], alpha=X_test[:,2]/X_test[:,2].max(), s=10*X_test[:,3], c=y_test, cmap='viridis')
ax.scatter(X_test[:,0], X_test[:,1], marker='+', c=y_hat, cmap='viridis')
ax.set_title('sklearn KNN\nTest data - transparency scaled based on '
             +wine.feature_names[2]+' and size scaled based on '+wine.feature_names[3]+'\n'
             +str(round(np.mean(y_hat == y_test)*100,2))+'% accuracy')
ax.set_xlabel(wine.feature_names[0])
ax.set_ylabel(wine.feature_names[1])
legend_elements = [Line2D([0], [0], marker='o', color='w', label=wine.target_names[0], markerfacecolor=cmap(0), markersize=10),
                   Line2D([0], [0], marker='o', color='w', label=wine.target_names[1], markerfacecolor=cmap(127), markersize=10),
                   Line2D([0], [0], marker='o', color='w', label=wine.target_names[2], markerfacecolor=cmap(255), markersize=10),
                   Line2D([0], [0], marker='P', color='w', label=wine.target_names[0]+' predicted', markerfacecolor=cmap(0), markersize=10),
                   Line2D([0], [0], marker='P', color='w', label=wine.target_names[1]+' predicted', markerfacecolor=cmap(127), markersize=10),
                   Line2D([0], [0], marker='P', color='w', label=wine.target_names[2]+' predicted', markerfacecolor=cmap(255), markersize=10)]
font = font_manager.FontProperties(size=6)
ax.legend(handles=legend_elements, loc='upper right', ncol=2, prop=font)
plt.show()