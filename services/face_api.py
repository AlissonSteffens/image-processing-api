from PIL import Image 
import numpy
import cv2
import services.simple_processing as simple_processing

def find_face(img):
    haar_cascade_face = cv2.CascadeClassifier('haar-models/haarcascade_frontalface_default.xml')
    faces_rects = haar_cascade_face.detectMultiScale(simple_processing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
    return faces_rects

def get_face(img):
    haar_cascade_face = cv2.CascadeClassifier('haar-models/haarcascade_frontalface_default.xml')
    faces_rects = haar_cascade_face.detectMultiScale(simple_processing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
    (x,y,w,h) = faces_rects[0]
    crop_img = img[y:y+h,x:x+w]
    return crop_img

def draw_face_rect(img):
    haar_cascade_face = cv2.CascadeClassifier('haar-models/haarcascade_frontalface_default.xml')
    faces_rects = haar_cascade_face.detectMultiScale(simple_processing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def draw_face_points(img):
    haar_cascade_face = cv2.CascadeClassifier('haar-models/haarcascade_eye.xml')
    faces_rects = haar_cascade_face.detectMultiScale(simple_processing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    haar_cascade_face = cv2.CascadeClassifier('haar-models/haarcascade_frontalface_default.xml')
    faces_rects = haar_cascade_face.detectMultiScale(simple_processing.to_gray_scale(img), scaleFactor = 1.2, minNeighbors = 5)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img