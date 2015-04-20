#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from loader import Face, AvgFace
import argparse
import yaml
import threading
import multiprocessing



def gen_avg(exconf):
    af = AvgFace(exconf)
    size = af.exconf["eyefitting"]["size"]
    total = np.zeros((size[0],size[1],3))
    total_score = 0
    n = len(af.faces)
    print("Total faces: ",n)
    for face in af.faces[:n]:
        fe = preproc(face.fit_eyes())
        total += fe
        cv2.imwrite("mapped/m_{}.jpg".format(face.image_id),fe)
    avg = total/n
    cv2.imwrite("avg.jpg",avg)

class BasicClassifier:
    def __init__(self):
        pass
    def fit(self,X,y): 
        X2 = np.array([rgb2gray(x) for x in X])
        X3 = [ np.histogram(np.ravel(local_binary_pattern(x,8,1,"uniform")),bins=10)[0] for x in X2 ]
        self.clf = KNeighborsClassifier()
        self.clf.fit(X3,y)

    def predict(self,X):
        X2 = np.array([rgb2gray(x) for x in X])
        X3 = [ np.histogram(np.ravel(local_binary_pattern(x,8,1,"uniform")),bins=10)[0] for x in X2 ]
        return self.clf.predict(X3)

def single_run(args):
    Classifier,X_train,X_test,y_train,y_test = args
    clf = Classifier()

    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    return float(sum(pred == y_test))/len(pred)
class MyClassifier:
    def __init__(self):
        pass
    def fit(self,X,y):
        n_components = len(X)-len(np.unique(y))-7
        self.ipca = IncrementalPCA(n_components=n_components, batch_size=None)
        #print("Fitting PCA and transforming")
        self.ipca.fit(X)
        X_pca = self.ipca.transform(X)

        self.lda = LDA()
        #print("Fitting LDA")
        self.lda.fit(X_pca,y)
        X_lda = self.lda.transform(X_pca)

        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(X_lda,y)

    def predict(self,X):
        X_pca = self.ipca.transform(X)
        X_lda = self.lda.transform(X_pca)
        return self.clf.predict(X_lda)


class Experiment(multiprocessing.Process):
    def __init__(self,exconf):
        multiprocessing.Process.__init__(self)
        self.exconf = exconf

    def run(self):
        print("Fitting eyes")
        self.af = AvgFace(self.exconf)
        self.X = np.array([np.ravel(face.fit_eyes()) for face in self.af.faces])
        print("End fit eyes")
        self.y = np.array([int(face.c)+1 for face in self.af.faces])
        assert len(np.unique(self.y)) >= 2

        print("Start xvalid")
        accs = list(self.xvalid(MyClassifier))
        self.queue.put(accs)
        print("End xvalid")

    def xvalid(self,Classifier):
        xvconf = self.exconf["crossvalidation"]
        p = multiprocessing.Pool(2)
        X = self.X
        y = self.y

        skf = StratifiedShuffleSplit(y, xvconf["k"], test_size=xvconf["test_size"],random_state=0)
        datas = [(Classifier,X[train_index],X[test_index],y[train_index],y[test_index]) for train_index,test_index in skf]
        r = p.map(single_run,datas)
        return r

        #for train_index,test_index in skf:
            #clf = Classifier()

            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]
            #clf.fit(X_train,y_train)
            #pred = clf.predict(X_test)
            #acc = float(sum(pred == y[test_index]))/len(pred)
            #yield acc

#@profile
def main():
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument("--config",type=str,metavar="EXCONF",required=True,help="YAML configuration")
    args = parser.parse_args()
    conf = yaml.load(open(args.config))
    exs = []
    queue = multiprocessing.Queue()
    for i,exconf in enumerate(conf["exs"]):
        ex = Experiment(exconf)
        ex.i = i
        ex.queue = queue
        exs.append(ex)
        ex.start()

    for ex in exs:
        ex.join()
        accs = queue.get()
        print(accs)
        print(np.mean(accs))

if __name__ == "__main__":
    main()
