import base64
import io
from PIL import Image
import numpy as np
import cv2

class ImageConverter:
    def __init__(self):
        pass

    @staticmethod
    def b64_to_cv(uri):
        img = uri.decode("utf-8").split(',')[1]
        msg = base64.b64decode(img)
        nparr = np.fromstring(msg, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img


    def process_pil(self,img, function):
        img = img.decode("utf-8").split(',')[1]
        msg = base64.b64decode(img)
        buf = io.BytesIO(msg)
        image_file = Image.open(buf)
        
        image_file = function(image_file)
        
        buf = io.BytesIO()
        image_file.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue())
        return 'data:image/png;base64,' + img_str.decode('utf-8')

    @staticmethod
    def cv_to_b64(image):
        retval, buffer = cv2.imencode('.png', image)
        img_str = base64.b64encode(buffer)
        return 'data:image/png;base64,' + img_str.decode('utf-8')

class SimpleImageProcessing:
    @staticmethod
    def to_gray_scale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def negative(img):
        return cv2.bitwise_not(img)

    @staticmethod
    def thumbnize(img):
        maxw = 400
        maxh = 300
        h, w, channels = img.shape
        if(w>h):
            maxh = int(h* maxw/w)
        if(h>w):
            maxw = int(w* maxh/h)
        maxsize = (maxw, maxh) 
        cv_image = cv2.resize(img, maxsize, interpolation= cv2.INTER_AREA )
        return cv_image
    
    @staticmethod
    def to_sepia(img):
        sepia = numpy.zeros((img.shape[0], img.shape[1], 3), dtype = numpy.float32)
        sepia[:, :, 0] = img[:, :, 0] * 0.272 + img[:, :, 1] * 0.534 + img[:, :, 2] * 0.131
        sepia[:, :, 1] = img[:, :, 0] * 0.349 + img[:, :, 1] * 0.686 + img[:, :, 2] * 0.168
        sepia[:, :, 2] = img[:, :, 0] * 0.393 + img[:, :, 1] * 0.769 + img[:, :, 2] * 0.189
        sepia[sepia > 255] = 255
        return sepia
    
    @staticmethod
    def sketch(img):
        SobelX = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        SobelX = cv2.convertScaleAbs(SobelX)
        SobelY = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        SobelY = cv2.convertScaleAbs(SobelY)
        sketch = cv2.add(SobelX, SobelY)
        sketch = SimpleImageProcessing.to_gray_scale(sketch)
        return cv2.bitwise_not(sketch)