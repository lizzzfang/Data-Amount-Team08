# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2018

@author: Yifei
"""

import numpy as np
import csv
import matplotlib.pylab as plt
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def plot(a,X,y):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(len(X-1)):
        ax.scatter(X[i], y, alpha=0.3, label=a[i])
        ax.set_xlabel('X')
        ax.set_ylabel('PM2.5')
        ax.set_title('Beijing PM2.5 Data Set')
    plt.legend(loc='upper right')
    fig.savefig('graph.png')

def plot_output(y_test,predictions):
    fig2, ax2 =plt.subplots(figsize=(12,8))
    ax2.scatter(y_test, predictions)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    fig2.savefig('output.png')

def plot_final(c,R,name):
    fig3, ax3 =plt.subplots(figsize=(12,8))
    ax3.plot(c, R)
    ax3.set_xlabel('Data Amount')
    ax3.set_ylabel(name)
    fig3.savefig('result/%s.png'%name)

def main():
    #load the data
    data = np.loadtxt('data.csv', delimiter=',', skiprows=1, usecols=range(0, 8))
    #load the header
    f = open('data.csv')
    reader = csv.reader(f)
    headers=next(reader,None)
    data2 = data.transpose()
    X = data[:,0:7]
    y = data[:,7]

    # plot each scatters of data
    a = headers  #headers
    plot(a,data2[1:7,:],data2[7,:])

    #init
    R = [];Mean = []

    # K-fold Cross Validation
    m = len(X)
    c = np.linspace(20, m , num=2000, endpoint=True, dtype=int)
    for i in c:
        data1=data[0:i]
        X1 = X[0:i]
        y1 = y[0:i]
        ss = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
        for train,test in ss.split(data1):
            #print('TRAIN:', train, 'TEST:', test)
            X_train, X_test, y_train, y_test = X1[train], X1[test], y1[train], y1[test]
        print('Test data/Train data: %s/%s' % (len(X_test),len(X_train)))


        #linear regression
        model = linear_model.BayesianRidge()
        model.fit(X_train, y_train)
        y_predictions = model.predict(X_test)

        #R^2 (coefficient of determination) regression score function
        r = r2_score(y_test, y_predictions)
        R.append(r)
        print('R-squared: %.4f' % r,end="        ")

        # Mean absolute error regression loss
        mean = mean_absolute_error(y_test, y_predictions)
        Mean.append(mean)
        print('Mean absolute error : %.4f' % mean)

    plot_final(c,R,'R^2')
    plot_final(c,Mean,'Mean absolute error')
    #plot the prediction
    #plot_output(y_test,y_predictions)




if __name__ == '__main__':
    main()
