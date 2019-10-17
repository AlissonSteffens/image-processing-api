import cv2
import numpy as np
import dlib
import simple_processing

def find_landmarks(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("roi_models/shape_predictor_68_face_landmarks.dat")
    gray = simple_processing.to_gray_scale(img)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
    return img
