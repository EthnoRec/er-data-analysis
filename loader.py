import numpy as np
import cv2
import os
import yaml
import requests
from StringIO import StringIO
from PIL import Image
import psycopg2
import psycopg2.extras
#import time

servers = yaml.load(open(os.environ["ER_SERVERS"]))


class Face:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.broken = False
        if "parent" in kwargs:
            self.exconf = self.parent.exconf
        assert hasattr(self, "exconf")

        if "image_id" in kwargs:
            self.filename = self.image_id + ".jpg"
            self.path = os.path.join(servers["image"]["dir"], self.filename)

            url_params = servers["image"]
            url_params["filename"] = self.filename
            self.url = "http://{host}:{port}/{filename}".format(**url_params)

    def crop_face(self, m):
        self.origin = self.bound[0]
        self.eye_left -= self.origin
        self.eye_right -= self.origin
        return m[
            self.bound[0][1]:self.bound[1][1],
            self.bound[0][0]:self.bound[1][0]]

    def download(self):
        r = requests.get(self.url)
        try:
            i = Image.open(StringIO(r.content))
            i.save(self.path)
        except:
            if os.path.exists(self.path):
                os.remove(self.path)

    def imread(self):
        if not os.path.exists(self.path):
            self.download()
        m = cv2.imread(self.path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #m = cv2.imread(self.path)
        if m is None:
            return None
        m = self.crop_face(m)
        return m

    def preproc(self, face):
        return face

    def fit_eyes(self):
        def midpoint(a, b):
            return np.mean([a, b], axis=0)

        def vector_cos(a, b):
            return np.dot(
                a, b)/(0.001+np.linalg.norm(a+0.001)*np.linalg.norm(b+0.001))

        size = tuple(self.exconf["eyefitting"]["size"])
        avg_eye_left = np.array([size[0]/2 - size[0]/5, size[1]*0.3])
        avg_eye_right = np.array([size[0]/2 + size[0]/5, size[1]*0.3])

        m = self.imread()
        if m is None:
            return None

        avg_eye_v = np.int16(avg_eye_right - avg_eye_left)
        eye_v = np.int16(self.eye_right - self.eye_left)

        angd = np.rad2deg(np.arccos(vector_cos(avg_eye_v, eye_v)))

        scale_f = np.linalg.norm(
            0.001+avg_eye_v)/(0.001+np.linalg.norm(0.001+eye_v))

        M = cv2.getRotationMatrix2D(
            tuple(
                midpoint(
                    self.eye_left, self.eye_right)), -angd, scale_f)
        M[:, 2] += midpoint(avg_eye_left, avg_eye_right) - \
            midpoint(self.eye_left, self.eye_right)

        m2 = cv2.warpAffine(m, M, size)
        #m3 = self.preproc(m2)
        return m2


class AvgFace:

    def __init__(self, exconf):
        self.exconf = exconf
        self.__avg_eyes()

    def get_query(self):
        def get_class_query(class_):
            keys = class_.keys()
            strs = ["{k} = '{v}'".format
                    (k=k, v=class_[k])
                    for k in["country", "city"] if k in keys]
            nums = ["{k} = {v}".format(k=k, v=class_[k])
                    for k in ["gender"] if k in keys]
            wheres = " AND ".join(strs+nums)

            class_query = """
                    (SELECT *,{id:d} as class FROM detection_eyes
                    JOIN \"People\" AS p ON p._id = person_id
                    JOIN \"Locations\" AS ls ON ls.lat = p.origin_lat AND ls.long = p.origin_long
                    WHERE {wheres}
                    ORDER BY score DESC LIMIT {n:d})
                    """.format(wheres=wheres, n=class_["n"], id=class_["id"])
            return class_query
        return " UNION ".join([get_class_query(class_)
                              for class_ in self.exconf["classes"]])

    def __avg_eyes(self):
        size = self.exconf["eyefitting"]["size"]
        with psycopg2.connect(**servers["database"]) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                q = self.get_query()

                #start_time = time.time()
                cur.execute(q)
                res = cur.fetchall()
                #end_time = time.time()
                #print("Fetched in {:2f}s".format(end_time - start_time))

                self.faces = []
                for row in res:
                    bound = [
                        [row["bound_origin_x"], row["bound_origin_y"]],
                        [row["bound_extent_x"], row["bound_extent_y"]]]
                    eye_left_m = np.mean(
                        [[row["left_origin_x"], row["left_origin_y"]],
                         [row["left_extent_x"], row["left_extent_y"]]], axis=0)
                    eye_right_m = np.mean(
                        [[row["right_origin_x"], row["right_origin_y"]],
                         [row["right_extent_x"], row["right_extent_y"]]],
                        axis=0)
                    self.faces.append(
                        Face(
                            parent=self,
                            row=row,
                            bound=bound,
                            image_id=row["image_id"],
                            eye_left=eye_left_m,
                            eye_right=eye_right_m,
                            c=row["class"]))
                self.avg_eye_left = np.array([size[0]/2-size[0]/10, size[1]/2])
                self.avg_eye_right = np.array([size[0]/2+size[0]/10, size[1]/2])
                self.faces = np.array(self.faces)
