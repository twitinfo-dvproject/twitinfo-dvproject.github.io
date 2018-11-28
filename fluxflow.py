"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.
"""
print(__doc__)

import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pandas as pd
import csv

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
#X = 0.3 * np.random.randn(100, 2)
X = pd.read_csv('dummy.csv',usecols=['ID'])
X_train = X.sample(frac=0.7)
#print('\nTest print:\n', X_train)
# Generate some regular novel observations
#X = 0.3 * np.random.randn(20, 2)
X_test = X.sample(frac=0.3)
# Generate some abnormal novel observations
#X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model

test = np.array(X_test)
train = np.array(X_train)
#test = scale(test)
pca= PCA(whiten=True)
pca.fit(test)
test = pca.transform(test)
clf = svm.OneClassSVM(kernel='rbf', degree=2, gamma=0.1, coef0=0.0,
                    tol=0.001, nu=0.1, shrinking=True, cache_size=200,
                    verbose=False, max_iter=-1, random_state=None)
clf.fit(train)
y_pred_train = clf.predict(train)
y_pred_test = clf.predict(test)
print('\n Y PRED TEST\n: ',y_pred_test)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

y_true = np.ones((13,1), dtype=np.int)
print(accuracy_score(y_true, y_pred_train))

# plot the line, the points, and the nearest vectors to the plane
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)



