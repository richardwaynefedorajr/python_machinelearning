from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def tuneHyperParameters(X_train, y_train):

    k_values = range(1,50)
    cv_k = lambda k: cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=10).mean()
    k_scores = list(map(cv_k, k_values))
    optimal_k = k_values[k_scores.index(max(k_scores))]
    plt.plot(k_values, k_scores)
    plt.title('K value vs. cross validation score: optimal k = '+str(optimal_k))
    plt.xlabel('K value')
    plt.ylabel('Accuracies')
    plt.savefig('knn_k_values.png')
    plt.show()
    
    return optimal_k