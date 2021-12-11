# ML for Gao's dataset 
# purpose: using SVM, K-nearest for "score" classificaiton (50 vs 60)
# April 25

import numpy as np
from sklearn import preprocessing, cross_validation,neighbors, svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import pandas as pd 

#define input files 
df = pd.read_csv('C:/Users/Ge Lan/Desktop/ML/gao.csv')
df.drop(['L1'], 1, inplace=True)

# identify independent(X)and dependent (y) variables 
X = np.array(df.drop(['Score'],1))
y = np.array(df['Score'])

# using cross_validation to train and test the model performance, test_size is set as 20% sample of the dataset
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.5)

# identify SVM, K-nearest
clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)

clf_k = neighbors.KNeighborsClassifier()
clf_k.fit(X_train, y_train)

accuracy_svm = clf_svm.score(X_test, y_test) 
accuracy_k = clf_k.score(X_test, y_test)

# evaluation result
print("SVM accuracy: ",accuracy_svm)
print("K-nearest accuracy: ",accuracy_k)
