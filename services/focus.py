import cv2
import json
import numpy as np
import skfuzzy as fuzz

class FocusFinder:
    def __init__(self):
        self.a = True

    def find_direction(self, marks, as_np = False):
        tb = []
        dist_boca_esq = abs(marks[1][0] - marks[5][0]) #x nariz menos x esq boca
        dist_boca_dir = abs(marks[1][0] - marks[4][0]) #x nariz - x direita boca
        
        dist_y_o_esq = abs(marks[1][1] - marks[3][1]) #y nariz - y esquerda olho
        dist_y_o_dir = abs(marks[1][1] - marks[2][1]) #y nariz - y direita olho

        dist_x_o_esq = abs(marks[1][0] - marks[3][0]) #x nariz - x esquerda olho
        dist_x_o_dir = abs(marks[1][0] - marks[2][0]) #x nariz - x direita olho

        dist_queixo = abs(marks[1][1] - marks[0][1]) #y nariz - y queixo


        dist_esq = (dist_x_o_esq + dist_boca_esq)/2
        dist_dir = (dist_x_o_dir + dist_boca_dir)/2

        dist_olhos = ((dist_y_o_esq+dist_y_o_dir)/2)
        percentual_lateral = dist_esq/(dist_esq+dist_dir)

        percentual_vertical = ((dist_y_o_esq+dist_y_o_dir)/2)/dist_queixo
        

        up = fuzz.membership.gaussmf(percentual_vertical,0,.3)
        middle = fuzz.membership.gaussmf(percentual_vertical,.5, .3)
        down = fuzz.membership.gaussmf(percentual_vertical,1.0,.25)

        

        right = fuzz.membership.gaussmf(percentual_lateral,0,.25)
        center = fuzz.membership.gaussmf(percentual_lateral,.5,.09)
        left = fuzz.membership.gaussmf(percentual_lateral,1,.25)
        
        if right > center and right > left:
            direction_h = 'Direita'
        else:
            if left > center and left > right:
                direction_h = 'Esquerda'
            else:
                direction_h = 'Frente'
        
        if up > middle and up > down:
            direction_v = 'Cima'
        else:
            if down > middle and down > up:
                direction_v = 'Baixo'
            else:
                direction_v = 'Reto' 

        tb.append({
            'distances':{
                'h_distance_nose_to_left_mouth':int(dist_boca_esq),
                'h_distance_nose_to_right_mouth':int(dist_boca_dir),
                'h_distance_nose_to_left_eye': int(dist_x_o_esq),
                'h_distance_nose_to_right_eye': int(dist_x_o_dir),
                'v_distance_nose_to_left_eye': int(dist_y_o_esq),
                'v_distance_nose_to_right_eye': int(dist_y_o_dir),
                'distance_nose_to_eyes': int(dist_olhos),
                'distance_nose_to_chin': int(dist_queixo),
            },
            'calculated' :{
                'x_rate': float(percentual_lateral),
                'y_rate': float(percentual_vertical),
            },          
            'fuzzy_is_looking':{
                'vertical':{
                    'up': up,
                    'middle': middle,
                    'down': down
                },
                'horizontal':{  
                    'right': right,
                    'center': center,
                    'left': left
                }
            },
            'direction_h': direction_h,
            'direction_v': direction_v
        })

        json_dump = json.dumps(tb)
        return json_dump