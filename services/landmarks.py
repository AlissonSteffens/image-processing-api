import cv2
import json
import numpy as np

class LandmarkFinder:
    def __init__(self):
        self.detector  = cv2.face.createFacemarkLBF()
        self.detector.loadModel('models/lbfmodel.yaml')

    def find_landmarks(self, image, faces, as_np = False):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        _, landmarks = self.detector.fit(image_gray, faces)

        if as_np:
            return landmarks

        tb = []
        count = 0
        for landmark in landmarks:
            table = []
            for x,y in landmark[0]:
                table.append({
                    'x':int(x),
                    'y':int(y)
                })
            tb.append({
                'face': count,
                'points': table
                })
            count+=1

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