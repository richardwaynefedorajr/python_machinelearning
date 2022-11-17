import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def train_by_method(method, X_train_in, y_train_in):
    return DecisionTreeClassifier(  criterion = method,
                                    random_state = 100,
                                    max_depth=3,
                                    min_samples_leaf=5 ).fit(X_train_in, y_train_in)
    
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    print("Report : ", classification_report(y_test, y_pred))

## import data
balance_data = pd.read_csv( 'https://archive.ics.uci.edu/ml/machine-learning-'+
                                'databases/balance-scale/balance-scale.data',
                                sep= ',', header = None)  
    
## split
X_train, X_test, y_train, y_test = train_test_split( balance_data.values[:, 1:5],
                                                     balance_data.values[:, 0],
                                                     test_size = 0.3,
                                                     random_state = 100 )

## train
clf_gini = train_by_method('gini', X_train, y_train)
clf_entropy = train_by_method('entropy', X_train, y_train)

## predictions
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)

## results
cal_accuracy(y_test, y_pred_gini)
cal_accuracy(y_test, y_pred_entropy)
