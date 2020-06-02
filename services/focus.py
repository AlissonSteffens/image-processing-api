import cv2
import json
import numpy as np
import skfuzzy as fuzz

class FocusFinder:
    def __init__(self):
        self.a = True

    def find_direction(self, marks, as_np = False):
        tb = []
        for face in marks:
            dist_esq = abs(face[1][0] - face[5][0]) #x nariz menos x esq boca
            dist_dir = abs(face[1][0] - face[4][0]) #x nariz - x direita boca
            
            dist_o_esq = abs(face[1][1] - face[3][1]) #y nariz - y esquerda olho
            dist_o_dir = abs(face[1][1] - face[2][1]) #y nariz - y direita olho
            dist_queixo = abs(face[1][1] - face[0][1]) #y nariz - y queixo


            dist_olhos = ((dist_o_esq+dist_o_dir)/2)
            percentual_lateral = dist_esq/(dist_esq+dist_dir)

            percentual_vertical = ((dist_o_esq+dist_o_dir)/2)/dist_queixo
            direita = False if dist_esq > dist_dir else True
            frente = True if  (percentual_lateral < .65 and percentual_lateral > .35) else False

            cima = True if percentual_vertical < .75 else False
            reto = True if  (percentual_vertical < .9 and percentual_vertical > .5) else False
            tb.append({
                'distances':{
                    'distance_nose_to_left_mouth':int(dist_esq),
                    'distance_nose_to_right_mouth':int(dist_dir),
                    'distance_nose_to_left_eye': int(dist_o_esq),
                    'distance_nose_to_rught_eye': int(dist_o_dir),
                    'distance_nose_to_eyes': int(dist_olhos),
                    'distance_nose_to_chin': int(dist_queixo),
                },
                'calculated' :{
                    'x_rate': float(percentual_vertical),
                    'y_rate': float(percentual_lateral),
                },          
                'fuzzy_is_looking':{
                    'vertical':{
                        'up': fuzz.membership.gaussmf(percentual_vertical,0,.25),
                        'middle': fuzz.membership.gaussmf(percentual_vertical,.8, .15),
                        'down': fuzz.membership.gaussmf(percentual_vertical,1,.25)
                    },
                    'horizontal':{  
                        'right': fuzz.membership.gaussmf(percentual_lateral,0,.25),
                        'center': fuzz.membership.gaussmf(percentual_lateral,.5,.1),
                        'left': fuzz.membership.gaussmf(percentual_lateral,1,.25)
                    }
                },
                'direction_h':'Frente' if frente else ('Esquerda' if direita else 'Direita'),
                'direction_v':'Reto' if reto else ('Cima' if cima else 'Baixo')
            })

        json_dump = json.dumps(tb)
        return json_dump