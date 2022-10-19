import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## load and split
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split( iris_data.data, iris_data.target, test_size = 0.3, random_state = 100 )

## train
model = LogisticRegression().fit(X_train, y_train)

## predict
y_hat = model.predict(X_test)

## accuracy
print(np.mean(y_hat == y_test))

## add data to plot
cmap = cm.get_cmap('viridis')
fig = plt.figure()
ax = plt.axes()
ax.scatter(X_test[:,0], X_test[:,1], alpha=X_test[:,2]/X_test[:,2].max(), s=100*X_test[:,3], c=y_test, cmap='viridis')
ax.scatter(X_test[:,0], X_test[:,1], marker='+', c=y_test, cmap='viridis')
ax.set_title('sklearn logistic regression\nTest data - transparency scaled based on '
             +iris_data.feature_names[2]+' and size scaled based on '+iris_data.feature_names[3]+'\n'
             +str(round(np.mean(y_hat == y_test)*100,2))+'% accuracy')
ax.set_xlabel(iris_data.feature_names[0])
ax.set_ylabel(iris_data.feature_names[1])
legend_elements = [Line2D([0], [0], marker='o', color='w', label=iris_data.target_names[0], markerfacecolor=cmap(0), markersize=10),
                   Line2D([0], [0], marker='o', color='w', label=iris_data.target_names[1], markerfacecolor=cmap(127), markersize=10),
                   Line2D([0], [0], marker='o', color='w', label=iris_data.target_names[2], markerfacecolor=cmap(255), markersize=10),
                   Line2D([0], [0], marker='P', color='w', label=iris_data.target_names[0]+' predicted', markerfacecolor=cmap(0), markersize=10),
                   Line2D([0], [0], marker='P', color='w', label=iris_data.target_names[1]+' predicted', markerfacecolor=cmap(127), markersize=10),
                   Line2D([0], [0], marker='P', color='w', label=iris_data.target_names[2]+' predicted', markerfacecolor=cmap(255), markersize=10)]
ax.legend(handles=legend_elements, loc='best', ncol=2)
plt.show()