from flask import Flask
from flask import request
import gray_scale
app = Flask(__name__)

@app.route('/to-gray-scale', methods=['POST'])
def get_image():
    imgBase64 = request.data
    return gray_scale.to_gray_scale(imgBase64)

if __name__ == '__main__':
    app.run(debug = True)