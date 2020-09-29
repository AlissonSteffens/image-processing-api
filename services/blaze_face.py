import math
import numpy as np

class SsdAnchorsCalculatorOptions:
    def __init__(self, input_size_width, input_size_height, min_scale, max_scale
                 , num_layers, feature_map_width, feature_map_height
                 , strides, aspect_ratios, anchor_offset_x=0.5, anchor_offset_y=0.5
                 , reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
                 , fixed_anchor_size=False):
        # Size of input images.
        self.input_size_width = input_size_width
        self.input_size_height = input_size_height
        # Min and max scales for generating anchor boxes on feature maps.
        self.min_scale = min_scale
        self.max_scale = max_scale
        # The offset for the center of anchors. The value is in the scale of stride.
        # E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
        self.anchor_offset_x = anchor_offset_x
        self.anchor_offset_y = anchor_offset_y
        # Number of output feature maps to generate the anchors on.
        self.num_layers = num_layers
        # Sizes of output feature maps to create anchors. Either feature_map size or
        # stride should be provided.
        self.feature_map_width = feature_map_width
        self.feature_map_height = feature_map_height
        self.feature_map_width_size = len(feature_map_width)
        self.feature_map_height_size = len(feature_map_height)
        # Strides of each output feature maps.
        self.strides = strides
        self.strides_size = len(strides)
        # List of different aspect ratio to generate anchors.
        self.aspect_ratios = aspect_ratios
        self.aspect_ratios_size = len(aspect_ratios)
        # A boolean to indicate whether the fixed 3 boxes per location is used in the lowest layer.
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer
        # An additional anchor is added with this aspect ratio and a scale
        # interpolated between the scale for a layer and the scale for the next layer
        # (1.0 for the last layer). This anchor is not included if this value is 0.
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio
        # Whether use fixed width and height (e.g. both 1.0f) for each anchor.
        # This option can be used when the predicted anchor width and height are in  pixels.
        self.fixed_anchor_size = fixed_anchor_size

    def to_string(self):
        return 'input_size_width: {:}\ninput_size_height: {:}\nmin_scale: {:}\nmax_scale: {:}\nanchor_offset_x: {:}\nanchor_offset_y: {:}\nnum_layers: {:}\nfeature_map_width: {:}\nfeature_map_height: {:}\nstrides: {:}\naspect_ratios: {:}\nreduce_boxes_in_lowest_layer: {:}\ninterpolated_scale_aspect_ratio: {:}\nfixed_anchor_size: {:}' \
            .format(self.input_size_width, self.input_size_height, self.min_scale, self.max_scale
                    , self.anchor_offset_x, self.anchor_offset_y, self.num_layers
                    , self.feature_map_width, self.feature_map_height, self.strides, self.aspect_ratios
                    , self.reduce_boxes_in_lowest_layer, self.interpolated_scale_aspect_ratio
                    , self.fixed_anchor_size)


class Anchor:
    def __init__(self, x_center, y_center, h, w):
        self.x_center = x_center
        self.y_center = y_center
        self.h = h
        self.w = w

    def to_string(self):
        return 'x_center: {:}, y_center: {:}, h: {:}, w: {:}'.format(self.x_center, self.y_center, self.h, self.w)


class Detection:
    def __init__(self, score, class_id, xmin, ymin, width, height):
        self.score = score
        self.class_id = class_id
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height

    def to_string(self):
        return 'score: {:}, class_id: {:}, xmin: {:}, ymin: {:}, width: {:}, height: {:}'.format(self.score,
                                                                                                 self.class_id,
                                                                                                 self.xmin, self.ymin,
                                                                                                 self.width,
                                                                                                 self.height)


class TfLiteTensorsToDetectionsCalculatorOptions:
    def __init__(self, num_classes, num_boxes, num_coords, keypoint_coord_offset
                 , ignore_classes, score_clipping_thresh, min_score_thresh
                 , num_keypoints=0, num_values_per_keypoint=2, box_coord_offset=0
                 , x_scale=0.0, y_scale=0.0, w_scale=0.0, h_scale=0.0, apply_exponential_on_box_size=False
                 , reverse_output_order=False, sigmoid_score=False, flip_vertically=False):
        # The number of output classes predicted by the detection model.
        self.num_classes = num_classes
        # The number of output boxes predicted by the detection model.
        self.num_boxes = num_boxes
        # The number of output values per boxes predicted by the detection model. The
        # values contain bounding boxes, keypoints, etc.
        self.num_coords = num_coords

        # The offset of keypoint coordinates in the location tensor.
        self.keypoint_coord_offset = keypoint_coord_offset
        # The number of predicted keypoints.
        self.num_keypoints = num_keypoints
        # The dimension of each keypoint, e.g. number of values predicted for each keypoint.
        self.num_values_per_keypoint = num_values_per_keypoint
        # The offset of box coordinates in the location tensor.
        self.box_coord_offset = box_coord_offset

        # Parameters for decoding SSD detection model.
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.w_scale = w_scale
        self.h_scale = h_scale

        self.apply_exponential_on_box_size = apply_exponential_on_box_size

        # Whether to reverse the order of predicted x, y from output.
        # If false, the order is [y_center, x_center, h, w], if true the order is
        # [x_center, y_center, w, h].
        self.reverse_output_order = reverse_output_order
        # The ids of classes that should be ignored during decoding the score for
        # each predicted box.
        self.ignore_classes = ignore_classes

        self.sigmoid_score = sigmoid_score
        self.score_clipping_thresh = score_clipping_thresh

        # Whether the detection coordinates from the input tensors should be flipped
        # vertically (along the y-direction). This is useful, for example, when the
        # input tensors represent detections defined with a coordinate system where
        # the origin is at the top-left corner, whereas the desired detection
        # representation has a bottom-left origin (e.g., in OpenGL).
        self.flip_vertically = flip_vertically

        # Score threshold for perserving decoded detections.
        self.min_score_thresh = min_score_thresh

    def to_string(self):
        return 'num_classes: {:}\nnum_boxes: {:}\nnum_coords: {:}\nkeypoint_coord_offset: {:}\nnum_keypoints: {:}\nnum_values_per_keypoint: {:}\nbox_coord_offset: {:}\nx_scale: {:}\ny_scale: {:}\nwx_scale: {:}\nh_scale: {:}\napply_exponential_on_box_size: {:}\nreverse_output_order: {:}\nignore_classes: {:}\nsigmoid_score: {:}\nscore_clipping_thresh: {:}\nflip_vertically: {:}\nmin_score_thresh: {:}' \
            .format(self.num_classes, self.num_boxes, self.num_coords, self.keypoint_coord_offset
                    , self.num_keypoints, self.num_values_per_keypoint, self.box_coord_offset
                    , self.x_scale, self.y_scale, self.w_scale, self.h_scale
                    , self.apply_exponential_on_box_size, self.reverse_output_order
                    , self.ignore_classes, self.sigmoid_score, self.score_clipping_thresh
                    , self.flip_vertically, self.min_score_thresh)


def DecodeBox(raw_boxes, i, anchors, options):
    box_data = np.zeros(options.num_coords)

    box_offset = i * options.num_coords + options.box_coord_offset

    y_center = raw_boxes[box_offset]
    x_center = raw_boxes[box_offset + 1]
    h = raw_boxes[box_offset + 2]
    w = raw_boxes[box_offset + 3]
    if (options.reverse_output_order):
        x_center = raw_boxes[box_offset]
        y_center = raw_boxes[box_offset + 1]
        w = raw_boxes[box_offset + 2]
        h = raw_boxes[box_offset + 3]

    x_center = x_center / options.x_scale * anchors[i].w + anchors[i].x_center
    y_center = y_center / options.y_scale * anchors[i].h + anchors[i].y_center

    if (options.apply_exponential_on_box_size):
        h = np.exp(h / options.h_scale) * anchors[i].h
        w = np.exp(w / options.w_scale) * anchors[i].w
    else:
        h = h / options.h_scale * anchors[i].h
        w = w / options.w_scale * anchors[i].w

    ymin = y_center - h / 2.0
    xmin = x_center - w / 2.0
    ymax = y_center + h / 2.0
    xmax = x_center + w / 2.0

    box_data[0] = ymin
    box_data[1] = xmin
    box_data[2] = ymax
    box_data[3] = xmax

    if (options.num_keypoints):
        for k in range(options.num_keypoints):
            offset = i * options.num_coords + options.keypoint_coord_offset + k * options.num_values_per_keypoint

            keypoint_y = raw_boxes[offset]
            keypoint_x = raw_boxes[offset + 1]
            if (options.reverse_output_order):
                keypoint_x = raw_boxes[offset]
                keypoint_y = raw_boxes[offset + 1]

            box_data[4 + k * options.num_values_per_keypoint] = keypoint_x / options.x_scale * anchors[i].w + anchors[
                i].x_center
            box_data[4 + k * options.num_values_per_keypoint + 1] = keypoint_y / options.y_scale * anchors[i].h + \
                                                                    anchors[i].y_center
    return box_data


def ConvertToDetections(raw_boxes, anchors_, detection_scores, detection_classes, options):
    output_detections = []
    for i in range(options.num_boxes):
        if (detection_scores[i] < options.min_score_thresh):
            # print('passed, score lower than threshold')
            continue
        # print("box_idx:{:}".format(i))
        box_offset = 0
        box_data = DecodeBox(raw_boxes, i, anchors_, options)
        detection = ConvertToDetection(
            box_data[box_offset + 0], box_data[box_offset + 1],
            box_data[box_offset + 2], box_data[box_offset + 3],
            detection_scores[i], detection_classes[i], options.flip_vertically)

        output_detections.append(detection);
    return output_detections


def ConvertToDetection(box_ymin, box_xmin, box_ymax, box_xmax, score, class_id, flip_vertically):
    # Detection detection;
    # detection.add_score(score);
    # detection.add_label_id(class_id);

    # LocationData* location_data = detection.mutable_location_data();
    # location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

    # LocationData::RelativeBoundingBox* relative_bbox = location_data->mutable_relative_bounding_box();

    # relative_bbox->set_xmin(box_xmin);
    # relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
    # relative_bbox->set_width(box_xmax - box_xmin);
    # relative_bbox->set_height(box_ymax - box_ymin);

    detection = Detection(score, class_id, box_xmin, (1.0 - box_ymax if flip_vertically else box_ymin),
                          (box_xmax - box_xmin), (box_ymax - box_ymin))

    # print('score: {:}, class_id: {:}\n, xmin: {:}, ymin: {:}, width: {:}, height: {:}'.format(score, class_id, box_xmin, (1.0 - box_ymax if flip_vertically else box_ymin), (box_xmax - box_xmin), (box_ymax - box_ymin)))

    return detection


def ProcessCPU(raw_boxes, raw_scores, anchors_, options):
    # Postprocessing on CPU for model without postprocessing op. E.g. output
    # raw score tensor and box tensor. Anchor decoding will be handled below.

    # boxes = DecodeBoxes(raw_boxes, anchors_, options)
    detection_scores = np.zeros(options.num_boxes)
    detection_classes = np.zeros(options.num_boxes)

    # Filter classes by scores.
    for i in range(options.num_boxes):
        class_id = -1
        max_score = np.finfo(float).min
        # Find the top score for box i.
        for score_idx in range(options.num_classes):
            # if (ignore_classes_.find(score_idx) == ignore_classes_.end()) {
            score = raw_scores[i * options.num_classes + score_idx]
            if options.sigmoid_score:
                if options.score_clipping_thresh > 0:
                    score = -options.score_clipping_thresh if score < -options.score_clipping_thresh else score
                    score = options.score_clipping_thresh if score > options.score_clipping_thresh else score
                score = 1.0 / (1.0 + np.exp(-score))
            if max_score < score:
                max_score = score
                class_id = score_idx
            # }
        detection_scores[i] = max_score
        detection_classes[i] = class_id

    output_detections = ConvertToDetections(raw_boxes, anchors_, detection_scores, detection_classes, options)
    return output_detections


def orig_nms(detections, threshold):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if len(detections) <= 0:
        return np.array([])
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    s = []
    for detection in detections:
        x1.append(detection.xmin)
        x2.append(detection.xmin + detection.width)
        y1.append(detection.ymin)
        y2.append(detection.ymin + detection.height)
        s.append(detection.score)
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    s = np.array(s)
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())  # read s using I

    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    return list(np.array(detections)[pick])


def gen_anchors(options):
    anchors = []
    # Verify the options.
    if options.strides_size != options.num_layers:
        print("strides_size and num_layers must be equal.")
        return []

    layer_id = 0
    while layer_id < options.strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < options.strides_size and options.strides[last_same_stride_layer] ==
               options.strides[layer_id]):
            scale = options.min_scale + (options.max_scale - options.min_scale) * 1.0 * last_same_stride_layer / (
                        options.strides_size - 1.0)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                for aspect_ratio_id in range(options.aspect_ratios_size):
                    aspect_ratios.append(options.aspect_ratios[aspect_ratio_id])
                    scales.append(scale)

                if options.interpolated_scale_aspect_ratio > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == options.strides_size - 1 else options.min_scale + (
                                options.max_scale - options.min_scale) * 1.0 * (last_same_stride_layer + 1) / (
                                                                                                            options.strides_size - 1.0)
                    scales.append(math.sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        for i in range(len(aspect_ratios)):
            ratio_sqrts = math.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        feature_map_height = 0
        feature_map_width = 0
        if options.feature_map_height_size > 0:
            feature_map_height = options.feature_map_height[layer_id]
            feature_map_width = options.feature_map_width[layer_id]
        else:
            stride = options.strides[layer_id]
            feature_map_height = math.ceil(1.0 * options.input_size_height / stride)
            feature_map_width = math.ceil(1.0 * options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    # TODO: Support specifying anchor_offset_x, anchor_offset_y.
                    x_center = (x + options.anchor_offset_x) * 1.0 / feature_map_width
                    y_center = (y + options.anchor_offset_y) * 1.0 / feature_map_height
                    w = 0
                    h = 0
                    if options.fixed_anchor_size:
                        w = 1.0
                        h = 1.0
                    else:
                        w = anchor_width[anchor_id]
                        h = anchor_height[anchor_id]
                    new_anchor = Anchor(x_center, y_center, h, w)
                    anchors.append(new_anchor)
        layer_id = last_same_stride_layer
    return anchors