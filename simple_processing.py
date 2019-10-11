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
    
