# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:41:24 2016

@author: ldy
"""



from time import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

from sklearn import svm
#
import numpy as np
acc = []
nums = [55, 60, 65, 70, 75, 80, 85]
for num in nums:
    X_train=np.load('feat_match/features%d_train.npy'%num)
    y_train=np.load('feat_match/label%d_train.npy'%num)
    X_test=np.load('feat_match/features%d_test.npy'%num)
    y_test=np.load('feat_match/label%d_test.npy'%num)

    print("Fitting the classifier to the training set")
    t0 = time()
    C = 1000.0  # SVM regularization parameter
    clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    
    print("Predicting...")
    t0 = time()
    y_pred = clf.predict(X_test)
    
    print "Accuracy: %.3f" %(accuracy_score(y_test, y_pred))
    acc.append(accuracy_score(y_test, y_pred))
print acc


