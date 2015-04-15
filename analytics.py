#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import psycopg2
import psycopg2.extras

con_str = "host=127.0.0.1 dbname=tinder_development user=tinder password=tinder_pw"
class Face:
    def __init__(self,**kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def imread(self):
        return cv2.imread("/home/lee/proj/tinder-gather/gather-images/"+self.image_id+".jpg")

class AvgFace:
    def __init__(self):
       self.__avg_eyes()
    def fit_eyes(self,face):
        def midpoint(a,b):
            return np.mean([a,b],axis=0)

        def vector_cos(a,b):
            return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        avg_eye_v = np.int16(self.avg_eye_right - self.avg_eye_left)
        eye_v = np.int16(face.eye_right - face.eye_left)

        angd = np.rad2deg(np.arccos(vector_cos(avg_eye_v,eye_v)))

        scale_f = np.linalg.norm(avg_eye_v)/np.linalg.norm(eye_v)

        M = cv2.getRotationMatrix2D(tuple(midpoint(face.eye_left,face.eye_right)),-angd,scale_f)
        M[:,2] += midpoint(self.avg_eye_left,self.avg_eye_right) - midpoint(face.eye_left,face.eye_right)

        m = face.imread()
        m2 = cv2.warpAffine(m,M,(128,128))
        #cv2.line(m2,tuple(self.avg_eye_left),tuple(self.avg_eye_right),(200,0,0),2)
        return m2
    def __avg_eyes(self):
        with psycopg2.connect(con_str) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                query = """
                        SELECT * FROM detection_eyes
                        JOIN \"People\" AS p ON p._id = person_id
                        WHERE component = 6 AND score > 0.5 AND
                        p.origin_lat = 20.659699;
                        """
                cur.execute(query)
                res = cur.fetchall()

                self.faces = []
                for row in res:
                    eye_left_m = np.mean([[row["left_origin_x"],row["left_origin_y"]],[row["left_extent_x"],row["left_extent_y"]]],axis=0)
                    eye_right_m = np.mean([[row["right_origin_x"],row["right_origin_y"]],[row["right_extent_x"],row["right_extent_y"]]],axis=0)
                    self.faces.append(Face(image_id=row["image_id"],eye_left=eye_left_m,eye_right=eye_right_m))
                self.avg_eye_left = np.array([64-12,64])
                self.avg_eye_right = np.array([64+12,64])

def gen_avg():
    af = AvgFace()
    total = np.zeros((128,128,3))
    total_score = 0
    n = len(af.faces)
    print("Total faces: ",n)
    for face in af.faces[:n]:
        fe = af.fit_eyes(face)
        total += fe
        cv2.imwrite("mapped/m_{}.jpg".format(face.image_id),fe)
    avg = total/n
    cv2.imwrite("avg.jpg",avg)


def main():
    gen_avg()
    #from sklearn.decomposition import IncrementalPCA
    #af = AvgFace()
    #faces = [np.ravel(af.fit_eyes(face)) for face in af.faces]
    ##print(len(faces))
    #n_components = 74
    #ipca = IncrementalPCA(n_components=n_components, batch_size=None)
    #ipca.fit(faces)
    #import pdb; pdb.set_trace()
    



if __name__ == "__main__":
    main()
