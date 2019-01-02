import numpy as np
import tensorflow as tf

import cv2
class TensoflowFaceDector():
    def __init__(self, mark_model='./frozen_face_ssd.pb'):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(mark_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def get_faceboxes(self, image,threshold):

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        confidences = []
        faceboxes=[]

        im_height = image.shape[0]
        im_width = image.shape[1]
        for index,result in enumerate(np.squeeze(boxes)):
            confidence = np.squeeze(scores)[index]
            if confidence > threshold:
                ymin, xmin, ymax, xmax = result
                x_left_top= int(xmin * im_width)
                y_left_top = int(ymin * im_height)
                x_right_bt = int(xmax * im_width)
                y_right_bt = int(ymax * im_height)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_top, y_left_top, x_right_bt, y_right_bt])
        return confidences,faceboxes

class MarkDetector:
    def __init__(self, mark_model='./frozen_face_GAN.pb'):
        self.face_detector =TensoflowFaceDector()
        self.cnn_input_size = 128
        self.marks = None
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(mark_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph
        self.sess = tf.Session(graph=detection_graph)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color)

    @staticmethod
    def move_box(box, offset):
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:
            return box
        elif diff > 0:
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):
        _, raw_boxes = self.face_detector.get_faceboxes(
            image=image, threshold=0.9)

        for box in raw_boxes:
            diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs(diff_height_width / 2))
            box_moved = self.move_box(box, [0, offset_y])
            facebox = self.get_square_box(box_moved)
            if self.box_in_image(facebox, image):
                return facebox

        return None

    def detect_marks(self, image_np):
        logits_tensor = self.graph.get_tensor_by_name('G/stack_1:0')
        predictions = self.sess.run(
            logits_tensor,
            feed_dict={'input_image_tensor:0': image_np})
        return predictions



class eyeDetector:
    def __init__(self, mark_model='./frozen_eye_design.pb'):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(mark_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph
        self.sess = tf.Session(graph=detection_graph)

    def detect_marks(self, image_np):
        logits_tensor = self.graph.get_tensor_by_name('ConvNet/stack:0')
        predictions = self.sess.run(
            logits_tensor,
            feed_dict={'input_image_tensor:0': image_np})
        return predictions

class gazeDetector:
    def __init__(self, mark_model='./frozen_gaze_design.pb'):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(mark_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph
        self.sess = tf.Session(graph=detection_graph)

    def detect_marks(self, image_np,pose):
        logits_tensor = self.graph.get_tensor_by_name('ConvNet/logits/BiasAdd:0')
        predictions = self.sess.run(
            logits_tensor,
            feed_dict={'input_image_tensor:0': image_np,'input_pose_tensor:0': pose})
        return predictions

