from flask import Flask
from flask import request
import simple_processing
import image_conversion
app = Flask(__name__)

@app.route('/to-gray-scale', methods=['POST'])
def transform_to_grayscale():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.to_gray_scale)


@app.route('/negative', methods=['POST'])
def negative_image():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.negative)

@app.route('/thumb', methods=['POST'])
def convert_to_thumb():
    imgBase64 = request.data
    return image_conversion.process(imgBase64, simple_processing.thumbnize)

if __name__ == '__main__':
    app.run(debug = True)