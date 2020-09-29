import cv2
import json
import numpy as np
from services.blaze_face import *
import time
import tflite
import tensorflow as tf
from tensorflow import keras

class FaceFinder:
    def find_faces(self, image, as_np = False):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        
        ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=128, input_size_height=128,
                                                                 min_scale=0.1484375, max_scale=0.75
                                                                 , anchor_offset_x=0.5, anchor_offset_y=0.5,
                                                                 num_layers=4
                                                                 , feature_map_width=[], feature_map_height=[]
                                                                 , strides=[8, 16, 16, 16], aspect_ratios=[1.0]
                                                                 , reduce_boxes_in_lowest_layer=False,
                                                                 interpolated_scale_aspect_ratio=1.0
                                                                 , fixed_anchor_size=True)

        anchors = gen_anchors(ssd_anchors_calculator_options)

        options = TfLiteTensorsToDetectionsCalculatorOptions(num_classes=1, num_boxes=896, num_coords=16
                                                            , keypoint_coord_offset=4, ignore_classes=[],
                                                            score_clipping_thresh=100.0, min_score_thresh=0.75
                                                            , num_keypoints=6, num_values_per_keypoint=2,
                                                            box_coord_offset=0
                                                            , x_scale=128.0, y_scale=128.0, w_scale=128.0, h_scale=128.0,
                                                            apply_exponential_on_box_size=False
                                                            , reverse_output_order=True, sigmoid_score=True,
                                                            flip_vertically=False)
        
        interpreter = tf.lite.Interpreter(model_path='models/face_detection_front.tflite')

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()


        img_height = image.shape[0]
        img_width = image.shape[1]

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        preprocess_start_time = time.time()
        # input shape
        input_width = input_details[0]["shape"][1]
        input_height = input_details[0]["shape"][2]
        # resize
        input_data = cv2.resize(img_rgb, (input_width, input_height)).astype(np.float32)
        # preprocess
        # input_data = (input_data)
        input_data = ((input_data - 127.5) / 127.5)
        # input_data = ((input_data)/255)
        input_data = np.expand_dims(input_data, axis=0)
        preprocess_end_time = time.time()
        inference_start_time = time.time()
        # set input data
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        regressors = interpreter.get_tensor(output_details[0]["index"])
        classificators = interpreter.get_tensor(output_details[1]["index"])
        inference_end_time = time.time()
        postprocess_start_time = time.time()
        raw_boxes = np.reshape(regressors, int(regressors.shape[0] * regressors.shape[1] * regressors.shape[2]))
        raw_scores = np.reshape(classificators, int(classificators.shape[0] * classificators.shape[1] * classificators.shape[2]))
        detections = ProcessCPU(raw_boxes, raw_scores, anchors, options)
        detections = orig_nms(detections, 0.85)

        if as_np:
            lista = []
            for detection in detections:
                x = int(img_width * detection.xmin)
                w = int(img_width * (detection.width))
                y = int(img_height * detection.ymin)
                h = int(img_height * (detection.height))
                lista.append([x,y,w,h])
            return lista
        
        tb = []
        for detection in detections:
            x = int(img_width * detection.xmin)
            w = int(img_width * (detection.width))
            y = int(img_height * detection.ymin)
            h = int(img_height * (detection.height))
            tb.append({
                'x':int(x),
                'y':int(y),
                'w':int(w),
                'h':int(h)
            })

        json_dump = json.dumps(tb)
        return json_dump