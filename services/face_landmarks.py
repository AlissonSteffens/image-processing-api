import cv2
import numpy as np
import dlib
from services.image_utils import ImageConverter, SimpleImageProcessing

class FaceLandmarker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/roi/shape_predictor_68_face_landmarks.dat")

    def find_landmarks(self, img):
        gray = SimpleImageProcessing.to_gray_scale(img)
        face = self.detector(gray)[0]
        
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks = self.predictor(gray, face)
        json = "["
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            json += "{x:"+str(x)+",y:"+str(y)+"},"
        json += "]"
        return json

    def draw_landmarks(self, img):
        gray = SimpleImageProcessing.to_gray_scale(img)
        faces = self.detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            landmarks = self.predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
        return img
