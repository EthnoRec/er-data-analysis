#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from loader import Face, AvgFace
import argparse
import yaml
import threading
import multiprocessing as mp
import time
import joblib



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

class Experiment:#(multiprocessing.Process):
    def __init__(self,exconf):
        #multiprocessing.Process.__init__(self)
        self.exconf = exconf

    def run(self):
        print("Fitting eyes")
        start_time = time.time()

        self.af = AvgFace(self.exconf)

        end_time = time.time()
        print("AvgFace done in {:2f}s".format(end_time - start_time))



        start_time = time.time()
        self.X = np.array([np.ravel(face.fit_eyes()) for face in self.af.faces])
        end_time = time.time()

        print("Fit eyes done in {:2f}s".format(end_time - start_time))

        print("End fit eyes")
        self.y = np.array([int(face.c)+1 for face in self.af.faces])
        assert len(np.unique(self.y)) >= 2

        print("Start xvalid")
        accs = list(self.xvalid(MyClassifier))
        self.queue.put(accs)
        print("End xvalid")

    def xvalid(self,Classifier):
        xvconf = self.exconf["crossvalidation"]
        #p = multiprocessing.Pool(2)
        X = self.X
        y = self.y

        skf = StratifiedShuffleSplit(y, xvconf["k"], test_size=xvconf["test_size"],random_state=0)
        #datas = [(Classifier,X[train_index],X[test_index],y[train_index],y[test_index]) for train_index,test_index in skf]
        #r = p.map(single_run,datas)
        #return r
        import pdb; pdb.set_trace()

        for i,(train_index,test_index) in enumerate(skf):
            clf = Classifier()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            start_time = time.time()

            clf.fit(X_train,y_train)

            after_train_time = time.time()

            pred = clf.predict(X_test)

            end_time = time.time()
            print("[xvalid] - #{}/{} trained in {:.2f}s, tested in {:.2f}s, total {:.2f}s"
                    .format(i+1,xvconf["k"],after_train_time - start_time,end_time - after_train_time,end_time - start_time))


            acc = float(sum(pred == y[test_index]))/len(pred)
            yield acc

from Queue import Queue
def action_xvalid(conf):
    exs = []
    queue = Queue()
    for i,exconf in enumerate(conf["exs"]):
        ex = Experiment(exconf)
        ex.i = i
        ex.queue = queue
        exs.append(ex)
        ex.run()

    for ex in exs:
        #ex.join()
        accs = queue.get()
        print(accs)
        print(np.mean(accs))
def action_avg(conf):
    pass



import sharedmem
import os
import psutil

def mem(pid):
    # return the memory usage in MB
    process = psutil.Process(pid)
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

q = mp.Queue()
class XValidator(mp.Process):
    """Parallel validator"""
    def __init__(self, X, y, train_index, test_index):
        mp.Process.__init__(self)
        self.X = np.frombuffer(X,dtype=np.uint8).reshape((len(y),200*200))
        self.y = np.frombuffer(y,dtype=np.uint8)
        self.train_index = train_index
        self.test_index = test_index
    def run(self):
        pid = os.getpid()
        print("[{}] - started with {:.2f}MB".format(pid,mem(os.getpid())))
        X_train, X_test = self.X[self.train_index], self.X[self.test_index]
        y_train, y_test = self.y[self.train_index], self.y[self.test_index]

        clf = MyClassifier()

        start_time = time.time()

        #print(X_train.shape)
        clf.fit(X_train,y_train)
        q.put(mem(os.getpid()))

        #after_train_time = time.time()

        #pred = clf.predict(X_test)

        #end_time = time.time()

        #print("[xvalid] - trained in {:.2f}s, tested in {:.2f}s, total {:.2f}s"
                #.format(after_train_time - start_time,end_time - after_train_time,end_time - start_time))


        #acc = float(sum(pred == y_test))/len(pred)
        
        
        

def main():
    w_start = time.time()
    n = 200
    size = (200,200)
    X = sharedmem.empty(n*size[0]*size[1],dtype=np.uint8)
    y = sharedmem.empty(n,dtype=np.uint8)
    X[:] = np.array(np.random.random_integers(0,255,n*size[0]*size[1]),dtype=np.uint8)
    y[:] = np.array(np.random.random_integers(0,2,n),dtype=np.uint8)

    #import pdb; pdb.set_trace()
    skf = StratifiedShuffleSplit(y, 8, test_size=0.2,random_state=0)
    validators = [XValidator(X, y, train_index, test_index) for train_index, test_index in skf]

    vgn = 1
    vgsize = 8
    vgroups = np.array(validators).reshape((vgn,vgsize))

    for vg in vgroups:
        for v in vg:
            v.start()

        t = 0
        for v in vg:
            v.join()
            t += q.get()
        print(t)

    
    w_end = time.time()
    print("[an] - total {:.2f}s"
            .format(w_end - w_start))

    return
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument("--config",type=str,metavar="EXCONF",required=True,help="YAML configuration")
    parser.add_argument("action",choices=["xvalid","avg"],help="Action to perform")
    args = parser.parse_args()
    conf = yaml.load(open(args.config))
    if args.action == "xvalid":
        action_xvalid(conf)
    elif args.action == "avg":
        action_avg(conf)
    else: 
        print("Unknown action")

if __name__ == "__main__":
    main()
