# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import keras
import numpy
import os
import numpy as np
import cv2
import services.face_api as face_api
import services.simple_processing as simple_processing


def get_emotion(img):
    keras.backend.clear_session()
    x=None
    y=None
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(simple_processing.to_gray_scale(face_api.get_face(img)), (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    #loading the model
    json_file = open('services/emotion/fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("services/emotion/fer.h5")
    yhat= loaded_model.predict(cropped_img)
    return labels[int(np.argmax(yhat))]