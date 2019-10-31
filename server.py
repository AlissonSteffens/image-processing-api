from flask import Flask
from flask import request
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2
import dlib
from services.face import FaceFinder
from services.face_landmarks import FaceLandmarker
from services.emotion import EmotionRecognition
from services.image_utils import ImageConverter, SimpleImageProcessing

app = Flask(__name__)
emotion_recognitor = EmotionRecognition()
face_landmarker = FaceLandmarker()
face_finder = FaceFinder()


@app.route('/grayscale', methods=['POST'])
def transform_to_grayscale():
    img = ImageConverter.b64_to_cv(request.data)
    img = SimpleImageProcessing.to_gray_scale(img)
    return ImageConverter.cv_to_b64(img)


@app.route('/sepia', methods=['POST'])
def transform_to_sepia():
    img = ImageConverter.b64_to_cv(request.data)
    img = SimpleImageProcessing.to_sepia(img)
    return ImageConverter.cv_to_b64(img)


@app.route('/negative', methods=['POST'])
def negative_image():
    img = ImageConverter.b64_to_cv(request.data)
    img = SimpleImageProcessing.negative(img)
    return ImageConverter.cv_to_b64(img)


@app.route('/thumb', methods=['POST'])
def convert_to_thumb():
    img = ImageConverter.b64_to_cv(request.data)
    img = SimpleImageProcessing.thumbnize(img)
    return ImageConverter.cv_to_b64(img)


@app.route('/sketch', methods=['POST'])
def convert_to_sketch():
    img = ImageConverter.b64_to_cv(request.data)
    img = SimpleImageProcessing.sketch(img)
    return ImageConverter.cv_to_b64(img)


@app.route('/draw-landmarks', methods=['POST'])
def draw_facelandmarks():
    img = ImageConverter.b64_to_cv(request.data)
    img = face_landmarker.draw_landmarks(img)
    return ImageConverter.cv_to_b64(img)


@app.route('/api/landmarks', methods=['POST'])
def find_facelandmarks():
    img = ImageConverter.b64_to_cv(request.data)
    return face_landmarker.find_landmarks(img)


@app.route('/api/emotion', methods=['POST'])
def get_face_emotion():
    img = ImageConverter.b64_to_cv(request.data)
    return emotion_recognitor.get_emotion(img)


if __name__ == '__main__':
    pass    
    

app.run(host='0.0.0.0',port=8080, debug=True)