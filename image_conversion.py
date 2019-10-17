import base64
import io
from PIL import Image
import numpy as np
import cv2

def data_uri_to_cv2_img(uri):
    img = uri.decode("utf-8").split(',')[1]
    msg = base64.b64decode(img)
    nparr = np.fromstring(msg, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img



def process_pil(img, function):
    img = img.decode("utf-8").split(',')[1]
    msg = base64.b64decode(img)
    buf = io.BytesIO(msg)
    image_file = Image.open(buf)
    
    image_file = function(image_file)
    
    buf = io.BytesIO()
    image_file.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue())
    return 'data:image/png;base64,' + img_str.decode('utf-8')

def process(img, function):
    image = function(data_uri_to_cv2_img(img))
    retval, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer)
    return 'data:image/png;base64,' + img_str.decode('utf-8')

def process_land(img, function, detector, predictor):
    image = function(data_uri_to_cv2_img(img),detector, predictor)
    retval, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer)
    return 'data:image/png;base64,' + img_str.decode('utf-8')

def process_to_json(img, function):
    output = function(data_uri_to_cv2_img(img))
    return output 
    