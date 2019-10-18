from flask import Flask
from flask import request
import dlib
import simple_processing
import image_conversion
import face_api
import face_landmarks

app = Flask(__name__)

detector = None
predictor = None

@app.route('/to-gray-scale', methods=['POST'])
def transform_to_grayscale():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.to_gray_scale)


@app.route('/negative', methods=['POST'])
def negative_image():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.negative)

@app.route('/face', methods=['POST'])
def find_faces():
    imgBase64 = request.data
    return image_conversion.process_to_json(imgBase64, face_api.find_face)

@app.route('/face-rect', methods=['POST'])
def draw_faces():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, face_api.draw_face)

@app.route('/face-points', methods=['POST'])
def draw_face_points():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, face_api.draw_face_points)


@app.route('/face-marks', methods=['POST'])
def draw_facelandmarks():
    imgBase64 = request.data
    return image_conversion.process_land(imgBase64, face_landmarks.find_landmarks, detector, predictor)

@app.route('/thumb', methods=['POST'])
def convert_to_thumb():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.thumbnize)

@app.route('/to-sepia', methods=['POST'])
def convert_to_sepia():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.to_sepia)

@app.route('/sketch', methods=['POST'])
def convert_to_sketch():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.sketch)

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("roi_models/shape_predictor_68_face_landmarks.dat")
    app.run(host='0.0.0.0',port=8080)