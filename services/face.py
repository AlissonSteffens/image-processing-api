from PIL import Image 
import numpy as np
import cv2
from services.image_utils import ImageConverter, SimpleImageProcessing

class FaceFinder:
    def __init__(self):
        self.frontal_face = cv2.CascadeClassifier('models/haar/haarcascade_frontalface_default.xml')

    def find_face(self, img):
        faces_rects = self.frontal_face.detectMultiScale(SimpleImageProcessing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
        return faces_rects

    def get_face(self, img):
        faces_rects = self.frontal_face.detectMultiScale(SimpleImageProcessing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
        (x,y,w,h) = faces_rects[0]
        crop_img = img[y:y+h,x:x+w]
        return crop_img
    
    def get_face_48(self, img):
        cropped_img = cv2.resize(SimpleImageProcessing.to_gray_scale(self.get_face(img)), (48, 48))
        return cropped_img

    def draw_face_rect(self, img):
        faces_rects = self.frontal_face.detectMultiScale(SimpleImageProcessing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
        for (x,y,w,h) in faces_rects:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img