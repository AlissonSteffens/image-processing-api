from PIL import Image 
import numpy
import cv2
import matplotlib.pyplot as plt


def to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def negative(img):
    return cv2.bitwise_not(img)

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

def to_sepia(img):
    
    sepia = numpy.zeros((img.shape[0], img.shape[1], 3), dtype = numpy.float32)
    sepia[:, :, 0] = img[:, :, 0] * 0.272 + img[:, :, 1] * 0.534 + img[:, :, 2] * 0.131
    sepia[:, :, 1] = img[:, :, 0] * 0.349 + img[:, :, 1] * 0.686 + img[:, :, 2] * 0.168
    sepia[:, :, 2] = img[:, :, 0] * 0.393 + img[:, :, 1] * 0.769 + img[:, :, 2] * 0.189
    sepia[sepia > 255] = 255
    ##sepia /= 255
    return sepia