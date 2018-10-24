'''
Created on Oct 19, 2018

@author: Saad

'''
import os
import pandas
import warnings
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

def plot(x,y,x_label='X-Axis',y_label='Y-Axis'):
    fig, ax =plt.subplots(figsize=(10,10))
    ax.scatter(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return (fig,ax)

def save_plot(fig,outfile):
    fig.savefig('%s.png'%outfile)

def main():
    SCRIPT_DIR = os.path.abspath(os.path.join(__file__,os.pardir))
    dataset_path = os.path.join(SCRIPT_DIR,os.pardir,'datasets','classification',
                       'dataset1','dataset1_formatted.csv')
    df = pandas.read_csv(dataset_path)
    array = df.values
    num_data_pts = len(array)
    step = int(num_data_pts/1000)

    train_sizes = list(range(step,num_data_pts,step))

    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(array[:,:-1],
                                                array[:,-1],train_size=train_size)
        model = LogisticRegression()
        model.fit(X_train,y_train)
        
        predictions = model.predict(X_test)

        print("---------------------")
        print("Train data size: %d." % X_train.shape[0])
        print("Test data size: %d." % X_test.shape[0])
        print("Accuracy: %.3f %%" % (metrics.accuracy_score(y_test, predictions)*100))
        print("Log loss: %.3f" % metrics.log_loss(y_test, predictions))
        print("---------------------")

if __name__=='__main__':
    main()
