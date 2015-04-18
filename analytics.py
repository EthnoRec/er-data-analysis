#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import psycopg2
import psycopg2.extras
from sklearn.decomposition import IncrementalPCA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.exposure import adjust_gamma, adjust_log
from skimage.exposure import rescale_intensity
from skimage.filter import gaussian_filter
import os
import yaml

config = yaml.load(open(os.environ["FDCONFIG"]))

size = (200,200)
class Face:
    def __init__(self,**kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def crop_face(self,m):
        self.origin = self.bound[0]
        self.eye_left -= self.origin
        self.eye_right -= self.origin
        return m[self.bound[0][1]:self.bound[1][1],self.bound[0][0]:self.bound[1][0]]
    def imread(self):
        m = cv2.imread(os.path.join(config["images"],self.image_id+".jpg"))
        m = self.crop_face(m)
        return m
    def fit_eyes(self):
        def midpoint(a,b):
            return np.mean([a,b],axis=0)
        def vector_cos(a,b):
            return np.dot(a,b)/(0.001+np.linalg.norm(a+0.01)*np.linalg.norm(b+0.01))

        avg_eye_left = np.array([size[0]/2 - size[0]/5,size[1]*0.3])
        avg_eye_right = np.array([size[0]/2 + size[0]/5,size[1]*0.3])

        m = self.imread()

        avg_eye_v = np.int16(avg_eye_right - avg_eye_left)
        eye_v = np.int16(self.eye_right - self.eye_left)

        angd = np.rad2deg(np.arccos(vector_cos(avg_eye_v,eye_v)))

        scale_f = np.linalg.norm(0.001+avg_eye_v)/(0.001+np.linalg.norm(0.001+eye_v))

        M = cv2.getRotationMatrix2D(tuple(midpoint(self.eye_left,self.eye_right)),-angd,scale_f)
        M[:,2] += midpoint(avg_eye_left,avg_eye_right) - midpoint(self.eye_left,self.eye_right)

        m2 = cv2.warpAffine(m,M,size)
        return m2

class AvgFace:
    def __init__(self):
       self.__avg_eyes()
    def __avg_eyes(self):
        with psycopg2.connect(**config["database"]) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                query = """
                        (SELECT *,0 as class FROM detection_eyes
                        JOIN \"People\" AS p ON p._id = person_id
                        WHERE component = 6 AND 
                        p.origin_lat = -19.916667 AND p.origin_long = -43.933333 --brazil
                        --p.origin_lat = 20.659699 AND p.origin_long = -103.349609 --guadalajara
                        ORDER BY score DESC LIMIT 500)
                        UNION

                        (SELECT *,1 as class FROM detection_eyes
                        JOIN \"People\" AS p ON p._id = person_id
                        WHERE component = 6 AND
                        p.origin_lat = 18.975 AND p.origin_long = 72.825833 --mumbai
                        ORDER BY score DESC LIMIT 500);
                        """
                cur.execute(query)
                res = cur.fetchall()

                self.faces = []
                for row in res:
                    bound = [[row["bound_origin_x"],row["bound_origin_y"]],[row["bound_extent_x"],row["bound_extent_y"]]]
                    eye_left_m = np.mean([[row["left_origin_x"],row["left_origin_y"]],[row["left_extent_x"],row["left_extent_y"]]],axis=0)
                    eye_right_m = np.mean([[row["right_origin_x"],row["right_origin_y"]],[row["right_extent_x"],row["right_extent_y"]]],axis=0)
                    self.faces.append(Face(row=row,bound=bound,image_id=row["image_id"],eye_left=eye_left_m,eye_right=eye_right_m,c=row["class"]))
                self.avg_eye_left = np.array([size[0]/2-size[0]/10,size[1]/2])
                self.avg_eye_right = np.array([size[0]/2+size[0]/10,size[1]/2])


def gen_avg():
    af = AvgFace()
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


def xvalid(Classifier,X,y):
    print(Classifier.__name__)
    accs = []
    skf = StratifiedShuffleSplit(y, 10, test_size=0.9,random_state=0)
    for train_index,test_index in skf:
        clf = Classifier()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        acc = float(sum(pred == y[test_index]))/len(pred)
        accs.append(acc)
        print("Accuracy: {}".format(acc))
    print("Mean accuracy: {}".format(np.mean(accs)))

pr = (15,85)
def preproc(face):
    def contrast_stretching(face):
        p2, p98 = np.percentile(face, pr)
        rescaled = rescale_intensity(face, in_range=(p2, p98))
        return rescaled

    def dog_filter(face,s1=1,s2=2):
        return gaussian_filter(face,s1) - gaussian_filter(face,s2)

    face = contrast_stretching(face)
    #g = adjust_log(rgb2gray(face),gain=2)
    #dog = dog_filter(g)
    #rescaled = rescale_intensity(dog,out_range=(0,255))
    #rescaled = contrast_eq(rescaled)
    #rescaled = rescale_intensity(g,out_range=(0,255))
    #print(rescaled.shape)
    return face
def main():
    #gen_avg()
    #return
    af = AvgFace()

    X = np.array([np.ravel(preproc(face.fit_eyes())) for face in af.faces])
    y = np.array([int(face.c)+1 for face in af.faces])
    print(np.unique(y))
    print(pr)
    xvalid(MyClassifier,X,y)

    #X = np.array([face for face in af.faces])
    #y = np.array([int(face.c)+1 for face in af.faces])
    #X2 = np.array([face.fit_eyes() for face in af.faces])
    #xvalid(BasicClassifier,X2,y)

if __name__ == "__main__":
    main()
