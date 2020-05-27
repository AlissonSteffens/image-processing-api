from flask import Flask, send_file
from flask import request
import os
import numpy as np
import cv2
import urllib.request as urllib
from io import BytesIO
from PIL import Image

from services.faces import FaceFinder
from services.landmarks import LandmarkFinder

# from services.emotion import EmotionRecognition
# from services.image_utils import ImageConverter, SimpleImageProcessing

app = Flask(__name__)

# emotion_recognitor = EmotionRecognition()
# face_landmarker = FaceLandmarker()
face_finder = FaceFinder()
landmark_finder =  LandmarkFinder()

def url_to_image(url):
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

def serve_pil_image(img):
    pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return img_io

@app.route('/api/face', methods=['GET'])
def get_face():
    img_url = request.args['image']
    image = url_to_image(img_url)
    faces = face_finder.find_faces(image)
    return faces

@app.route('/api/marks', methods=['GET'])
def get_marks():
    img_url = request.args['image']
    image = url_to_image(img_url)
    faces = face_finder.find_faces(image,as_np = True)
    marks = landmark_finder.find_landmarks(image, faces)
    return marks

@app.route('/draw-marks', methods=['GET'])
def draw_marks():
    if 'image' in request.args:
        img_url = request.args['image']
    else:
        img_url = 'https://raw.githubusercontent.com/AlissonSteffens/image-processing-api/master/demo/lenna.jpg'

    if 'size' in request.args:
        size = int(request.args['size'])
    else:
        size = 1


    image = url_to_image(img_url)
    faces = face_finder.find_faces(image,as_np = True)
    marks = landmark_finder.find_landmarks(image, faces,as_np = True)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for landmark in marks:
        for x,y in landmark[0]:
            cv2.circle(image_rgb, (x, y), 1, (255, 255, 255), size)

    return send_file(serve_pil_image(image_rgb),mimetype='image/jpeg')
    

if __name__ == '__main__':
    pass    
    

app.run(host='0.0.0.0',port=8080, debug=True)