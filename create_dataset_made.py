import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_dataset(n_samples=100, n_features=100):
	X,y = make_classification(n_samples = n_samples, n_features = n_features, n_clusters_per_class=2,  random_state = 1)
	data = pd.DataFrame(np.append(X,y[:,np.newaxis],axis = 1))

	data.to_csv('made_dataset.csv')

	print ('Created dataset with shape of ({0},{1}) successfully'.format(n_samples,n_features))
	
	return

make_dataset(n_samples=1000500)
