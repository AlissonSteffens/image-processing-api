from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json, load_model
import keras
import numpy
import os
import numpy as np
import cv2
from services.faces import FaceFinder


class EmotionRecognition:
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self):
        self.andsa = True;

    def get_emotion(self, img):
        dicti = {}
        keras.backend.clear_session()
        cropped_img = np.expand_dims(np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1), 0)

        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        self.loaded_model = load_model("models/keras/emotion.h5")
        yhat = self.loaded_model.predict(cropped_img)

        count = 0
        for l in self.labels:
            dicti[l] = str("{:.5f}".format((yhat[0][count])))
            count+=1 
        
        dicti['main_emotion']= self.labels[np.argmax(yhat[0])]

        return dicti