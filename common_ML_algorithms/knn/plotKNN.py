import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def plotKNN(input_data, X_test, y_test, y_hat, optimal_k):
        
    ## create dataframe for visualization
    df = pd.DataFrame(data=np.c_[input_data['data'], input_data['target']], columns=input_data['feature_names'] + ['target'])
    pd.options.display.max_columns = None
    df.info()
    print(df.describe())
    
    ## histograms
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    sns.histplot(data=df, x=input_data.feature_names[0], bins=50, color="chartreuse", ax=axs[0, 0])
    sns.histplot(data=df, x=input_data.feature_names[1], bins=50, color="lawngreen", ax=axs[0, 1])
    sns.histplot(data=df, x=input_data.feature_names[2], bins=50, color="green", ax=axs[1, 0])
    sns.histplot(data=df, x=input_data.feature_names[3], bins=50, color="lime", ax=axs[1, 1])
    plt.suptitle('Histograms of iris dataset features')
    fig.tight_layout()
    plt.savefig('knn_histograms.png', bbox_inches='tight')
    plt.show()
    
    ## correlation matrix
    ax = sns.heatmap(df[input_data.feature_names].corr(), cmap="Greens", annot=True, square=True)
    ax.set_ylim(0, len(df[input_data.feature_names].corr()))
    plt.title('Correlation matrix for iris dataset features', y=1.05)
    plt.tight_layout()
    plt.savefig('knn_corr_matrix.png', bbox_inches='tight')
    plt.show()
    
    ## pairplots
    df['target'][df["target"] == 0.0] = input_data.target_names[0]
    df['target'][df["target"] == 1.0] = input_data.target_names[1]
    df['target'][df["target"] == 2.0] = input_data.target_names[2]
    sns.pairplot(df, hue="target", palette=['greenyellow','lime', 'darkgreen'], markers=["o", "s", "D"])
    plt.suptitle('Pair plots for features of iris dataset', y=1.025)
    plt.savefig('knn_pairplots.png', bbox_inches='tight')
    plt.show()
    
    ## prediction pairplots -> evaluate accuracy
    y_hat[y_hat != y_test] = 3.0
    df_pred = pd.DataFrame(data=np.c_[X_test, y_hat], columns=input_data['feature_names'] + ['target'])

    df_pred['target'][df_pred["target"] == 0.0] = input_data.target_names[0]
    df_pred['target'][df_pred["target"] == 1.0] = input_data.target_names[1]
    df_pred['target'][df_pred["target"] == 2.0] = input_data.target_names[2]
    df_pred['target'][df_pred["target"] == 3.0] = 'wrongly classified'

    sns.pairplot(df_pred, hue="target", palette=['greenyellow','lime', 'darkgreen', 'red'], markers=["o", "s", "D", "X"])
    plt.suptitle('KNN results: k = '+str(optimal_k)+' with accuracy: '+str(round(metrics.accuracy_score(y_test, y_hat)*100, 2)+'%'), y=1.025)
    plt.savefig('knn_pairplots_predicted.png', bbox_inches='tight')
    plt.show()