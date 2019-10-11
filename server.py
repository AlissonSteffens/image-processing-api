from flask import Flask
from flask import request
import simple_processing
import image_conversion
import face_api
app = Flask(__name__)

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

@app.route('/thumb', methods=['POST'])
def convert_to_thumb():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.thumbnize)

if __name__ == '__main__':
    app.run(debug = True)