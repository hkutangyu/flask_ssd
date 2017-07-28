import tensorflow as tf
from PIL import Image
import numpy as np
import json
from functools import wraps
import time


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_category_index(path_to_labels):
    category_index = {}
    with open(path_to_labels, mode='r', encoding='utf-8') as f:
        labels = json.load(f)
        for index, label in enumerate(labels):
            category_index[index+1] = {'id': index+1, 'name': label}
    return category_index


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" % ('function', str(t1-t0)))
        return result
    return function_timer


class ObjectDetectSsd:
    def __init__(self, path_to_ckpt):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


    def detect_single_image(self, image_path):

        with tf.Session(graph=self.detection_graph) as sess:

            image = Image.open(image_path)

            # numpy format of image
            image_np = load_image_into_numpy_array(image)
            t0 = time.time()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            t1 = time.time()

            print("Total time running %s: %s seconds" % ('function', str(t1 - t0)))
            return boxes, scores, classes, num_detections

