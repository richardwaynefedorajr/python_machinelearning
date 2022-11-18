import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## load, split, and preprocess data
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, test_size = 0.1, random_state = 100 )

# massage data for numpy solution
y_train_np = y_train.reshape((-1,1))
y_test_np = y_test.reshape((-1,1))
X_train_np = np.concatenate([X_train, np.ones_like(y_train_np, dtype=np.float32)], axis=1)
X_test_np = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)], axis=1)

## train
sk_model = LogisticRegression().fit(X_train, y_train)

if X_train_np.shape[0] >= X_train_np.shape[1] == np.linalg.matrix_rank(X_train_np):
    y_train_np = np.maximum(1e-5, np.minimum(y_train_np.astype(np.float32), 1-1e-5))
    # w = [(X^T*T)^-1*X^T]*log(1/y) -1
    weights =  np.matmul( np.matmul( np.linalg.inv( np.matmul( X_train_np.transpose(), X_train_np) ),
               X_train_np.transpose()), -np.log(np.divide(1, y_train_np) - 1))
else:
    print('X does not have full column rank')
    weights = 0
    
## predict
y_hat_sk = sk_model.predict(X_test)
y_hat_np = np.divide(1, 1+np.exp(-np.matmul(X_test_np, weights)))
zeros, ones = np.zeros_like(y_hat_np), np.ones_like(y_hat_np)
y_hat_np = np.where(y_hat_np >= 0.5, ones, zeros)

## plot and evaluate
## classify errors: 2 = np solution misclassified, 3 = sk misclassified, 4 = both misclassifed
def returnClassificationResults(test, sk, np):
    if test == sk & test == np:
        return test
    elif test != sk & test == np:
        return 2
    elif test == sk & test != np:
        return 3
    else:
        return 4
    
y_plot = np.array(list(map(returnClassificationResults, y_test, y_hat_sk, y_hat_np)))

# dataframe for seaborn plot
cancer_df = pd.DataFrame(np.c_[X_test, y_plot], columns= np.append(cancer['feature_names'], ['target']))
cancer_df = cancer_df.sort_values(by=['target'])
cancer_df['target'][cancer_df["target"] == 0.0] = 'Correctly classifed as '+cancer.target_names[0]
cancer_df['target'][cancer_df["target"] == 1.0] = 'Correctly classifed as '+cancer.target_names[1]
cancer_df['target'][cancer_df["target"] == 2.0] = 'wrongly classified by numpy solution'
cancer_df['target'][cancer_df["target"] == 3.0] = 'wrongly classified by scikit-learn solution'
cancer_df['target'][cancer_df["target"] == 4.0] = 'wrongly classified by both numpy and \nscikit-learn solutions'

sns.pairplot(cancer_df, hue='target', palette=['lime', 'darkgreen', 'red', 'darkred', 'crimson'], markers=['D', 'D', 'X', 'X', 'X'], 
             vars=cancer.feature_names[6:10])
plt.suptitle('Logistic regression results: numpy '+str(round(np.mean(y_hat_np == y_test_np)*100, 2))+'%, scikit-learn '+
             str(round(np.mean(y_hat_sk == y_test)*100, 2))+'%', y=1.025)
plt.savefig('logistic_regression_pairplots_predicted.png', bbox_inches='tight')
plt.show()