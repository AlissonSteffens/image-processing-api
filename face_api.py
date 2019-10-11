from PIL import Image 
import numpy
import cv2
import matplotlib.pyplot as plt
import simple_processing

def find_face(img):
    haar_cascade_face = cv2.CascadeClassifier('haar-models/haarcascade_frontalface_default.xml')
    faces_rects = haar_cascade_face.detectMultiScale(simple_processing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
    return 'Faces found: ' + str(faces_rects)