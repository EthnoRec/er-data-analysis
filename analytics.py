#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from loader import Face, AvgFace
import argparse
import yaml
import multiprocessing as mp
import time
import os
import psutil



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

class MyClassifier:
    def __init__(self):
        pass
    def fit(self,X,y):
        pass
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

class Experiment:
    def __init__(self,exconf):
        self.exconf = exconf
        # Confusion matrix queue
        self.cmq = mp.Queue()

    def make_validator(self, train_index, test_index):
        xv = XValidator()
        xv.faces_train, xv.y_train = self.af.faces[train_index], self.y[train_index]
        xv.faces_test, xv.y_test = self.af.faces[test_index], self.y[test_index]
        xv.cmq = self.cmq
        return xv

    def run(self):
        start_time = time.time()

        self.af = AvgFace(self.exconf)
        self.y = np.array([int(face.c)+1 for face in self.af.faces])
        assert len(np.unique(self.y)) >= 2

        end_time = time.time()
        print("AvgFace done in {:2f}s".format(end_time - start_time))


        xvconf = self.exconf["crossvalidation"]


        skf = StratifiedShuffleSplit(self.y, xvconf["k"], test_size=xvconf["test_size"],random_state=0)
        validators = [self.make_validator(train_index,test_index) for train_index, test_index in skf]


        vgsize = xvconf["vgsize"]
        assert xvconf["k"]%vgsize == 0
        vgn = xvconf["k"]/vgsize

        vgroups = np.array(validators).reshape((vgn,vgsize))

        n_classes = len(np.unique(self.y))
        cm = np.zeros((n_classes,n_classes))
        for vg in vgroups:
            for v in vg:
                v.start()

            for v in vg:
                v.join()

        while not self.cmq.empty():
            cm += self.cmq.get()
        print(cm)

def action_xvalid(conf):
    exs = []
    queue = mp.Queue()
    for i,exconf in enumerate(conf["exs"]):
        ex = Experiment(exconf)
        exs.append(ex)
        ex.run()

def action_avg(conf):
    pass


def mem(pid):
    # return the memory usage in MB
    process = psutil.Process(pid)
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

class XValidator(mp.Process):
    """Parallel validator"""
    def __init__(self):
        mp.Process.__init__(self)
    def run(self):
        pid = os.getpid()
        print("[{}] - started with {:.2f}MB ({}/{} samples)".format(pid,mem(os.getpid()),len(self.y_train),len(self.y_test)))

        self.X_train = np.array([np.ravel(face.fit_eyes()) for face in self.faces_train])
        self.X_test = np.array([np.ravel(face.fit_eyes()) for face in self.faces_test])
        print("[{}] - fit eyes {:.2f}MB".format(pid,mem(os.getpid())))

        clf = MyClassifier()

        start_time = time.time()

        clf.fit(self.X_train,self.y_train)
        print("[{}] - fit CLF {:.2f}MB".format(pid,mem(os.getpid())))

        after_train_time = time.time()

        y_pred = clf.predict(self.X_test)

        end_time = time.time()

        print("[{}] - trained in {:.2f}s, tested in {:.2f}s, total {:.2f}s"
                .format(pid,after_train_time - start_time,end_time - after_train_time,end_time - start_time))

        self.cmq.put(confusion_matrix(self.y_test,y_pred))
        
        
def main():
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument("--config",type=str,metavar="EXCONF",required=True,help="YAML configuration")
    parser.add_argument("action",choices=["xvalid","avg"],help="Action to perform")
    args = parser.parse_args()
    conf = yaml.load(open(args.config))
    w_start = time.time()
    if args.action == "xvalid":
        action_xvalid(conf)
    elif args.action == "avg":
        action_avg(conf)
    else: 
        print("Unknown action")
    w_end = time.time()
    print("[an] - total {:.2f}s"
            .format(w_end - w_start))

if __name__ == "__main__":
    main()
