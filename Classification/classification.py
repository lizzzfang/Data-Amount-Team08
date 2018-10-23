# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:49:29 2018

@author: Shuqi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation


def classfication(data):
    clf = KNeighborsClassifier()
    array = data.values
    X = data[:,0:5]
    y = data[:,6]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
    clf.fit(X_train,y_train)       
    predictions = clf.predict(X_test)

    print("Accuracy: %.3f %%" % (clf.accuracy_score(y_test, predictions)*100))
    print("Log loss: %.3f" % clf.log_loss(y_test, predictions))
    print("---------------------")



def main():
     #load the data
     df = pd.read_csv('dataset1_formatted.csv',header=None,names=['age','fnlwgt','education','capital gain','capital loss','hours per week','salary status'])
     X = data[:,0:5]
     y = data[:,6]
     
     #
     classfication(df)