from PIL import Image 
import numpy
import cv2


def to_gray_scale(img):
    cv_image = numpy.array(img)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(cv_image)

def negative(img):
    cv_image = numpy.array(img)
    cv_image = cv2.bitwise_not(cv_image)
    return Image.fromarray(cv_image)

def thumbnize(img):
    maxw = 400
    maxh = 300
    cv_image = numpy.array(img)

    h, w, channels = cv_image.shape
    if(w>h):
        maxh = int(h* maxw/w)
    if(h>w):
        maxw = int(w* maxh/h)

    maxsize = (maxw, maxh) 
    
    cv_image = cv2.resize(cv_image, maxsize, interpolation= cv2.INTER_AREA )
    return Image.fromarray(cv_image)
    
