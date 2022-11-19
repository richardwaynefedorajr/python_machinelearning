import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import graphviz

def train_by_method(method, X_train_in, y_train_in):
    return DecisionTreeClassifier(  criterion = method,
                                    random_state = 100,
                                    max_depth=3,
                                    min_samples_leaf=5 ).fit(X_train_in, y_train_in)

def getGraphviz(input_classifer, criterion_name):
    dot_data = export_graphviz(input_classifer, out_file='tree_'+criterion_name+'.dot', 
                                    feature_names=['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'], 
                                    class_names=['L', 'B', 'R'],
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph
    
# def cal_accuracy(y_test, y_pred):
#     print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
#     print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
#     print("Report : ", classification_report(y_test, y_pred))

## import data
balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', 
                           sep= ',', header = None)  
balance_data.rename(columns={1:'Left-Weight', 2:'Left-Distance', 3:'Right-Weight', 4:'Right-Distance'}, inplace=True)

## split
X_train, X_test, y_train, y_test = train_test_split(balance_data.values[:,1:5], balance_data.values[:,0],
                                                    test_size = 0.3, random_state = 100)

## train
clf_gini = train_by_method('gini', X_train, y_train)
clf_entropy = train_by_method('entropy', X_train, y_train)

## predictions
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)

## results
print("Accuracy Gini criterion: "+str(round(accuracy_score(y_test,y_pred_gini)*100, 2))+'%')
print("Accuracy Entropy criterion: "+str(round(accuracy_score(y_test,y_pred_entropy)*100, 2))+'%')

# cal_accuracy(y_test, y_pred_gini)
# cal_accuracy(y_test, y_pred_entropy)

## visualize tree -> dot -Tpng tree.dot -o tree.png    (PNG format)
getGraphviz(clf_gini, 'gini')
getGraphviz(clf_entropy, 'entropy')