import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plotKNN(input_data):
    
    ## create dataframe for visualization
    df = pd.DataFrame(data=np.c_[input_data['data'], input_data['target']], columns=input_data['feature_names'] + ['target'])
    
    ## histograms
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    sns.histplot(data=df, x=input_data.feature_names[0], bins=50, color="chartreuse", ax=axs[0, 0])
    sns.histplot(data=df, x=input_data.feature_names[1], bins=50, color="lawngreen", ax=axs[0, 1])
    sns.histplot(data=df, x=input_data.feature_names[2], bins=50, color="green", ax=axs[1, 0])
    sns.histplot(data=df, x=input_data.feature_names[3], bins=50, color="lime", ax=axs[1, 1])
    fig.tight_layout()
    plt.savefig('knn_histograms.png')
    plt.show()
    
    ## correlation matrix
    ax = sns.heatmap(df[input_data.feature_names].corr(), cmap="Greens", annot=True, square=True)
    ax.set_ylim(0, len(df[input_data.feature_names].corr()))
    plt.tight_layout()
    plt.savefig('knn_corr_matrix.png')
    plt.show()
    
    ## pairplots
    df['target'][df["target"] == 0.0] = input_data.target_names[0]
    df['target'][df["target"] == 1.0] = input_data.target_names[1]
    df['target'][df["target"] == 2.0] = input_data.target_names[2]
    sns.pairplot(df, hue="target")
    plt.savefig('knn_pairplots.png')
    plt.show()