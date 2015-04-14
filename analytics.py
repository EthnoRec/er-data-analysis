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
        m2 = cv2.warpAffine(m,M,(540,540))
        #cv2.line(m2,tuple(self.avg_eye_left),tuple(self.avg_eye_right),(200,0,0),2)
        return m2
    def __avg_eyes(self):
        def midpoints(boxes):
            xs = 0.5*(boxes[:,2] + boxes[:,0])
            ys = 0.5*(boxes[:,3] + boxes[:,1])
            return np.uint16([xs,ys]).transpose()

        with psycopg2.connect(con_str) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT min(origin_x) AS origin_x, min(origin_y) \
                        AS origin_y,max(extent_x) AS extent_x, max(extent_y) AS extent_y, \
                        fd.image_id AS image_id \
                        FROM \"Boxes\" AS bs JOIN \"FaceDetections\" as fd \
                        ON fd._id = bs.fd_id WHERE part_index between 9 and 14 AND fd.component = 6 AND score > 0.7 \
                        GROUP BY fd._id,fd.image_id;")
                eye_left_rs = cur.fetchall()
                eye_left_data = np.array(eye_left_rs)

                # Midpoints of left-eye boxes (merged by PostgreSQL)
                left_eyes = midpoints(np.uint16(eye_left_data[:,:-1]))

                image_ids = eye_left_data[:,-1]

                # Average of these midpoints
                self.avg_eye_left = np.uint16(left_eyes.mean(axis=0))

                cur.execute("SELECT min(origin_x) as origin_x, min(origin_y) \
                        AS origin_y,max(extent_x) AS extent_x, max(extent_y) AS extent_y \
                        FROM \"Boxes\" AS bs JOIN \"FaceDetections\" as fd \
                        ON fd._id = bs.fd_id WHERE part_index between 20 and 25 AND fd.component = 6 AND score > 0.7 \
                        GROUP BY fd._id,fd.image_id;")
                eye_right_rs = cur.fetchall()

                # Midpoints of right-eye boxes (merged by PostgreSQL)
                right_eyes = midpoints(np.array(eye_right_rs))

                self.avg_eye_right = np.uint16(right_eyes.mean(axis=0))
                self.faces = [Face(eye_left=eye_left,eye_right=eye_right,image_id=image_id) for eye_left,eye_right,image_id in zip(left_eyes,right_eyes,image_ids)]

def main():


    af = AvgFace()
    total = np.zeros((540,540,3))
    for face in af.faces:
        fe = af.fit_eyes(face)
        total += fe
        #cv2.imwrite("mapped/m_{}.jpg".format(face.image_id),fe)
    avg = total/len(af.faces)
    cv2.imwrite("avg.jpg",avg)



if __name__ == "__main__":
    main()
