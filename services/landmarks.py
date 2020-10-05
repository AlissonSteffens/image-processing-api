import cv2
import json
import numpy as np
import tflite
import tensorflow as tf
from tensorflow import keras
import time

class LandmarkFinder:

    def find_landmarks(self, image, offset = (0,0), as_np = False):
        interpreter = tf.lite.Interpreter(model_path='models/face_landmark.tflite')
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        floating_model = input_details[0]['dtype'] == np.float32
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        img = cv2.resize(image,(width, height))
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 127.5) /127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        results.shape = (468,3)

        for i in range(len(results)):
            results[i][0] = results[i][0]*(image.shape[1]/width)+offset[0]
            results[i][1] = results[i][1]*(image.shape[0]/height)+offset[1]
        if as_np:
            return results

        tb = []
        table = []
        for x,y,z in results:
            table.append({
                'x':int(x),
                'y':int(y),
                'z':int(z),
            })
        tb.append({
            'face': 'face',
            'points': table
        })

        json_dump = json.dumps(tb)
        return json_dump
