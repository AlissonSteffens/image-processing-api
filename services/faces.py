import cv2
import json
import numpy as np

class FaceFinder:
    def __init__(self):
        self.detector = cv2.CascadeClassifier('models/haar/haarcascade_frontalface_default.xml')
    
    
    def find_faces(self, image, as_np = False):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(image_gray)

        if as_np:
            return faces

        tb = []
        for f in faces:
            (x,y,w,h) = f
            tb.append({
                'x':int(x),
                'y':int(y),
                'w':int(w),
                'h':int(h)
            })

        json_dump = json.dumps(tb)
        return json_dump