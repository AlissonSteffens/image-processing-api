# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json, load_model
import keras
import numpy
import os
import numpy as np
import cv2
from services.image_utils import ImageConverter, SimpleImageProcessing
from services.face import FaceFinder


class EmotionRecognition:
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self):
        self.face_finder = FaceFinder()

    def get_emotion(self, img):
        keras.backend.clear_session()
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(SimpleImageProcessing.to_gray_scale(self.face_finder.get_face(img)), (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        self.loaded_model = load_model("models/keras/emotion.h5")
        yhat= self.loaded_model.predict(cropped_img)
        return self.labels[int(np.argmax(yhat))]