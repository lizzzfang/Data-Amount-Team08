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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics

def k_fold(data):
    X = data.iloc[:,0:5]
    y = data.iloc[:,6]    
    floder = KFold(n_splits=4,random_state=0,shuffle=False)
    #KFold
    for train, test in floder.split(X,y):
        print('Train: %s | test: %s' % (train, test))
        print(" ")
    


def S_fold(data):
    X = data.iloc[:,0:5]
    y = data.iloc[:,6]       
    sfolder = StratifiedKFold(n_splits=4,random_state=0,shuffle=False)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    for train,test in sfolder.split(X1,y1):
       print('Train: %s | test: %s' % (train, test))
       

            
def plot_output(y_test,predictions):
    fig2, ax2 =plt.subplots(figsize=(12,8))
    ax2.scatter(y_test, predictions)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    fig2.savefig('output.png')     
  


   
def main():
     #load the data
     df = pd.read_csv('dataset1_formatted.csv',header=None,names=['age','fnlwgt','education','capital gain','capital loss','hours per week','salary status'])        
     #randome apply sample
     #DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
# =============================================================================
#      for i in np.arange(0.1,1,0.1):
#          data=df.sample(frac=i,axis=0)
#          k_fold(data)
# =============================================================================
         
     # K_fold Cross Validation 
     #k_fold(df)
# =============================================================================
#      m = len(X)
#      c = np.linspace(20, m , num=2000, endpoint=True, dtype=int)
#      for i in c:
#          X1 = X[0:i]
#          y1 = y[0:i]
#          kf= KFold(n_splits=4,random_state=0,shuffle=False)
#          kf.get_n_splits(X1)
#          for train,test in kf.split(X1):
#              #print('Train: %s | test: %s' % (train, test))
#              X_train,X_test=X[train],X[test]
#              y_train,y_test=y[train],y[test]
#              model = LogisticRegression() 
#              model.fit(X_train, y_train)       
#              predictions = model.predict((X_test))
#          #plot_output(y_test,predictions)
#          
# 
#      fig, ax = plt.subplots(figsize=(12,8))
#      ax.scatter(y_test, predictions)
#      ax.set_xlabel('True Values')
#      ax.set_ylabel('Predictions')
#      fig.savefig('output.png')     
#      
#      
# =============================================================================

     array = df.values
     num_data_pts = len(array)
     step = int(num_data_pts/1000)
     for test_data_size in range(num_data_pts-step,step,-step):
         X_train, X_test, y_train, y_test = model_selection.train_test_split(array[:,:-1],
                                                 array[:,-1],test_size=test_data_size)
         model = LogisticRegression()
         model.fit(X_train,y_train)
         
         predictions = model.predict(X_test)

         print("---------------------")
         print("Train data size: %d." % X_train.shape[0])
         print("Test data size: %d." % X_test.shape[0])
         print("Accuracy: %.3f %%" % (metrics.accuracy_score(y_test, predictions)*100))
         print("Accuracy2: %.3f %%" % (metrics.average_precision_score(y_test, predictions)*100))
         print("Log loss: %.3f" % metrics.log_loss(y_test, predictions))
         print("---------------------")
         
         #plot_output(y_test,predictions)
         
       
     # StratifiedKFold Cross Validation
     #S_fold(df)   
     
     
if __name__ == '__main__':
    main()     
     
     
     
     
     
     
     