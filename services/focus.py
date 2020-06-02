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
            dist_boca_esq = abs(face[1][0] - face[5][0]) #x nariz menos x esq boca
            dist_boca_dir = abs(face[1][0] - face[4][0]) #x nariz - x direita boca
            
            dist_y_o_esq = abs(face[1][1] - face[3][1]) #y nariz - y esquerda olho
            dist_y_o_dir = abs(face[1][1] - face[2][1]) #y nariz - y direita olho

            dist_x_o_esq = abs(face[1][0] - face[3][0]) #x nariz - x esquerda olho
            dist_x_o_dir = abs(face[1][0] - face[2][0]) #x nariz - x direita olho

            dist_queixo = abs(face[1][1] - face[0][1]) #y nariz - y queixo


            dist_esq = (dist_x_o_esq + dist_boca_esq)/2
            dist_dir = (dist_x_o_dir + dist_boca_dir)/2

            dist_olhos = ((dist_y_o_esq+dist_y_o_dir)/2)
            percentual_lateral = dist_esq/(dist_esq+dist_dir)

            percentual_vertical = ((dist_y_o_esq+dist_y_o_dir)/2)/dist_queixo
            

            up = fuzz.membership.gaussmf(percentual_vertical,0,.25)
            middle = fuzz.membership.gaussmf(percentual_vertical,.8, .15)
            down = fuzz.membership.gaussmf(percentual_vertical,1,.25)

            

            right = fuzz.membership.gaussmf(percentual_lateral,0,.25)
            center = fuzz.membership.gaussmf(percentual_lateral,.5,.06)
            left = fuzz.membership.gaussmf(percentual_lateral,1,.25)
            
            if right > center and right > left:
                direction_h = 'Right'
            else:
                if left > center and left > right:
                    direction_h = 'Left'
                else:
                    direction_h = 'Center'
            
            if up > middle and up > down:
                direction_v = 'Up'
            else:
                if down > middle and down > up:
                    direction_v = 'Down'
                else:
                    direction_v = 'Middle' 

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