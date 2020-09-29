import cv2
import json
import numpy as np
import tflite
import tensorflow as tf
from tensorflow import keras
import time

class LandmarkFinder:

    def find_landmarks(self, image, as_np = False):
        interpreter = tf.lite.Interpreter(model_path='models/face_landmark.tflite')
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()


        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        print(width,height)
        img = cv2.resize(image,(width, height))

        # add N dim
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

        # for i in range(len(results)):
        #     results[i][0] = results[i][0]*(image.shape[1]/width)
        #     results[i][1] = results[i][1]*(image.shape[0]/height)
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
    
    def find_simple_landmarks(self, image, faces, as_np = False):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        _, landmarks = self.detector.fit(image_gray, faces)

        pontos = []

        
        pontos.append(9) # queixo
        pontos.append(34) # nariz
        pontos.append(37) # direita do olho direito
        pontos.append(46) # esquerda do olho esquerdo
        pontos.append(49) # direita boca
        pontos.append(55) # esquerda boca
        


        if as_np:
            tb = []
            count = 0
            for landmark in landmarks:
                table = []
                indice = 1
                for x,y in landmark[0]:
                    if indice in pontos:
                        table.append([int(x),int(y)])
                    indice+=1
                tb.append(table)
                count+=1
                
            return tb

        tb = []
        count = 0
        for landmark in landmarks:
            table = []
            indice = 1
            for x,y in landmark[0]:
                if indice in pontos:
                    table.append({
                        'x':int(x),
                        'y':int(y)
                    })
                indice+=1
            tb.append({
                'face': count,
                'points': table
                })
            count+=1

        json_dump = json.dumps(tb)
        return json_dump