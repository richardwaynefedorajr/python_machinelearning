# Deep Learning
Examples of common methodologies in deep learning, implemented in pytorch and tensorflow

## Linear Regression
Visualizations generated from pytorch and tensorflow implementations.

| ![pytorch linear regression output](./linear_regression/linear_regression_torch.svg) |
|:--:| 
| *Pytorch linear regression output* |

| ![tensorflow linear regression output](./linear_regression/linear_regression_tensorflow.svg) |
|:--:| 
| *Tensorflow linear regression output* |

## MNIST Fashion Dataset - Softmax Classifier, MLP, and CNN

Comparison of classification results on MNIST Fashion dataset between pytorch softmax classifier, tensorflow MLP, and pytorch CNN.

First, the softmax results:

| ![pytorch softmax regression metrics](./softmax_regression/softmax_training_metrics.svg) |
|:--:| 
| *Pytorch softmax classifier training metrics* |

| ![pytorch softmax regression output](./softmax_regression/softmax_confusion_matrix.svg) |
|:--:| 
| *Pytorch softmax classifier results* |

Let us extend this line of action (switching to tensorflow for implementation), and add a hidden layer...

| ![tensorflow_MLP metrics](./MLP/MLP_training_metrics.svg) |
|:--:| 
| *Tensorflow MLP training metrics* |

| ![tensorflow_MLP output](./MLP/MLP_confusion_matrix.svg) |
|:--:| 
| *Tensorflow MLP results* |


Finally, a CNN implementation in pytorch (generally follows LeNet architecture, but substititutes relu activation function and adds batch normalization)

| ![pytorch CNN metrics](./CNN/CNN_training_metrics.svg) |
|:--:| 
| *Pytorch CNN training metrics* |

| ![pytorch CNN output](./CNN/CNN_confusion_matrix.svg) |
|:--:| 
| *Pytorch CNN results* |

Displayed below are examples of mis-labeled elements of the validation dataset:

| ![pytorch CNN mislabeled](./CNN/CNN_mislabeled.svg) |
|:--:| 
| *Pytorch CNN mis-labeled image examples - correct label above, incorrect label below* |