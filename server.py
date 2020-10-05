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
from services.focus import FocusFinder
from services.emotion import EmotionRecognition

# from services.emotion import EmotionRecognition
# from services.image_utils import ImageConverter, SimpleImageProcessing

app = Flask(__name__)

# emotion_recognitor = EmotionRecognition()
# face_landmarker = FaceLandmarker()
face_finder = FaceFinder()
landmark_finder =  LandmarkFinder()
focus_finder = FocusFinder()
emotion_recognitor = EmotionRecognition()

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

@app.route('/api/emotion', methods=['GET'])
def get_emotion():
    size = 48
    img_url = request.args['image']
    image = url_to_image(img_url)
    faces = face_finder.find_faces(image)
    faces = face_finder.find_faces(image, as_np = True)

    (x,y,w,h) = faces[0]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_img = image_rgb[y:y+h, x:x+w]

    final_image = cv2.resize(crop_img,(size,size))

    emotion = emotion_recognitor.get_emotion(final_image)

    return emotion


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
    xf,yf,wf,hf = faces[0]
    marginx = int(wf/2)
    marginy = int(hf/2)
    crope = image[max(int(yf-marginy),0):min(int(yf+hf+marginy),image.shape[0]), max(int(xf-marginx),0):min(int(xf+wf+marginx),image.shape[1])]
    marks = landmark_finder.find_landmarks(crope, offset=(max(int(xf-marginx),0),max(int(yf-marginy),0)))

    return marks

@app.route('/api/direction', methods=['GET'])
def get_direction():
    img_url = request.args['image']

    image = url_to_image(img_url)
    faces = face_finder.find_faces(image,as_np = True)
    marks = landmark_finder.find_simple_landmarks(image, faces,as_np = True)
    direction = focus_finder.find_direction(marks)

    return direction

@app.route('/face', methods=['GET'])
def get_face_rect():
    if 'image' in request.args:
        img_url = request.args['image']
    else:
        img_url = 'https://raw.githubusercontent.com/AlissonSteffens/image-processing-api/master/demo/lenna.jpg'

    image = url_to_image(img_url)
    
    faces = face_finder.find_faces(image, as_np = True)

    (x,y,w,h) = faces[0]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_img = image_rgb[y:y+h, x:x+w]

    if 'image_size' in request.args:
        size = int(request.args['image_size'])
        final_image = cv2.resize(crop_img,(size,size))
        return send_file(serve_pil_image(final_image),mimetype='image/jpeg')
    else:
        return send_file(serve_pil_image(crop_img),mimetype='image/jpeg')

@app.route('/face-marks', methods=['GET'])
def draw_face_marks():
    if 'image' in request.args:
        img_url = request.args['image']
    else:
        img_url = 'https://raw.githubusercontent.com/AlissonSteffens/image-processing-api/master/demo/lenna.jpg'

    if 'marker_size' in request.args:
        marker_size = int(request.args['marker_size'])
    else:
        marker_size = 1

    image = url_to_image(img_url)
    faces = face_finder.find_faces(image,as_np = True)
    xf,yf,wf,hf = faces[0]
    marginx = int(wf/2)
    marginy = int(hf/2)
    crope = image[max(int(yf-marginy),0):min(int(yf+hf+marginy),image.shape[0]), max(int(xf-marginx),0):min(int(xf+wf+marginx),image.shape[1])]
    marks = landmark_finder.find_landmarks(crope,as_np = True)
    image_rgb = cv2.cvtColor(crope, cv2.COLOR_BGR2RGB)
    
    for x,y,z in marks:
        pox = int(x)
        poy =  int(y)
        image_rgb[poy, pox] = (255, 255, 255)
        # cv2.circle(image_rgb, (pox, poy), 1, (255, 255, 255), marker_size)

    return send_file(serve_pil_image(image_rgb),mimetype='image/jpeg')  

if __name__ == '__main__':
    pass    
    

app.run(host='0.0.0.0',port=8080, debug=True)