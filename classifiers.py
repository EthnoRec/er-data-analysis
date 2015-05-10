import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.lda import LDA
from skimage.feature import local_binary_pattern

class BasicClassifier:
    def __init__(self,options):
        for key in options:
            setattr(self,key,options[key])
        self.n_points = 8*self.radius
    def fit(self,X,y):
        X3 = [ np.histogram(np.ravel(local_binary_pattern(x,self.n_points,self.radius,"uniform")),bins=self.bins)[0] for x in X ]
        self.clf = KNeighborsClassifier()
        self.clf.fit(X3,y)

    def predict(self,X):
        X3 = [ np.histogram(np.ravel(local_binary_pattern(x,self.n_points,self.radius,"uniform")),bins=self.bins)[0] for x in X ]
        return self.clf.predict(X3)


