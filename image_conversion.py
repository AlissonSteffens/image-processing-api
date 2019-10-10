import base64
import io
from PIL import Image 


def process(img, function):
    img = img.decode("utf-8").split(',')[1]
    msg = base64.b64decode(img)
    buf = io.BytesIO(msg)
    image_file = Image.open(buf)
    
    image_file = function(image_file)
    
    buf = io.BytesIO()
    image_file.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue())
    return 'data:image/png;base64,' + img_str.decode('utf-8')