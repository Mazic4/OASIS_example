import numpy as np
import pandas as pd


def create_dataset(n_instances,n_testing,random_state=1):
        df = pd.read_csv('made_dataset.csv',nrows=n_instances)
        data = df.values
        data = data[:,1:]
        data = data.astype(int)
        data[data[:,-1]==0,-1]=2
        np.random.seed(random_state)
        np.random.shuffle(data)

        Xtrain,ytrain = data[:n_instances-n_testing,:-1],data[:n_instances-n_testing,-1]
        Xtest,ytest = data[n_instances-n_testing:n_instances,:-1],data[n_instances-n_testing:n_instances,-1]

        return Xtrain,ytrain,Xtest,ytest