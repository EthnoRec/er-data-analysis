import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.lda import LDA
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

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


class PCALDA:
    def __init__(self,options):
        for key in options:
            setattr(self,key,options[key])
    def fit(self,X,y):
        X.resize((X.shape[0],X.shape[1]*X.shape[2]))
        if not hasattr(self,"pca_dim"):
            self.pca_dim = len(X)-len(np.unique(y))
        self.ipca = IncrementalPCA(n_components=self.pca_dim, batch_size=None)
        self.ipca.fit(X)
        
        X_pca = self.ipca.transform(X)

        self.lda = LDA()
        assert len(X_pca) == len(y)
        self.lda.fit(X_pca,y)
        X_lda = self.lda.transform(X_pca)

        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(X_lda,y)

    def predict(self,X):
        X.resize((X.shape[0],X.shape[1]*X.shape[2]))
        X_pca = self.ipca.transform(X)
        X_lda = self.lda.transform(X_pca)
        return self.clf.predict(X_lda)

