# -*- coding: utf-8 -*-

import os, pdb
import numpy as np
import tensorflow as tf
from PIL import Image

from models.model import Model
from box import Box

from .yolo_v3 import yolo_v3
from .yolo_v3_tiny import yolo_v3_tiny
from .utils import load_coco_names, draw_boxes, draw_boxes2, \
    get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, load_graph, letter_box_image

dir_ = os.path.dirname(__file__)

class Yolov3(Model):
    devices = ['CPU', 'GPU']

    def __init__(self, args):
        if args.device not in self.devices:
            exit(0)

        args.data_format = 'NHWC',  # Data format: NCHW (gpu only) / NHWC
        fname = 'yolov3_tiny' if args.tiny else 'yolov3'
        args.class_names = f"{dir_}/coco.names"  # File with class names
        args.weights_file = f"{dir_}/{fname}.weights"  # Binary file with detector weights
        args.ckpt_file = f"{dir_}/saved_model/{fname}.ckpt"  # Checkpoint file
        args.frozen_model = f"{dir_}/{fname}.pb"  # Frozen tensorflow protobuf model
        args.size = 416  # Image size
        args.conf_threshold = 0.7  # Confidence threshold
        args.iou_threshold = 0.6  # IoU threshold
        args.gpu_memory_fraction = 1.0  # Gpu memory fraction to use

        # FIXME init network in __init__, re-use here
        if args.device == 'CPU':
            # args.frozen_model = 'models/tensorflow_yolo_v3/frozen_darknet_yolov3_model_CPU.pb'
            # args.data_format = 'NHWC'
            config = tf.ConfigProto(device_count={'GPU': 0})
        elif args.device == 'GPU':
            # args.frozen_model = 'models/tensorflow_yolo_v3/frozen_darknet_yolov3_model_GPU.pb'
            # args.data_format = 'NCHW'
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            config = tf.ConfigProto(
                gpu_options=gpu_options,
                log_device_placement=False,
            )
        else:
            raise NotImplementedError()

        self.classes = load_coco_names(args.class_names)
        if args.frozen_model:
            frozenGraph = load_graph(args.frozen_model)
            self.boxes, self.inputs = get_boxes_and_inputs_pb(frozenGraph)
            self.sess = tf.Session(graph=frozenGraph, config=config)
        else:
            model = yolo_v3_tiny if args.tiny else yolo_v3()
            self.boxes, self.inputs = get_boxes_and_inputs(model, len(self.classes), args.size, args.data_format)
            saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
            self.sess = tf.Session(config=config)
            saver.restore(self.sess, args.ckpt_file)

        super().__init__(args)

    def get_bboxes(self, frame):
        args = self.args

        img = Image.fromarray(np.uint8(frame))  # .fromarray(frame)
        img_resized = letter_box_image(img, args.size, args.size, 128)
        img_resized = img_resized.astype(np.float32)

        detected_boxes = self.sess.run(
            self.boxes,
            feed_dict={self.inputs: [img_resized]}
        )

        filtered_boxes = non_max_suppression(
            detected_boxes,
            confidence_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )

        return draw_boxes2(filtered_boxes, img, self.classes, (args.size, args.size), True)

    def close(self):
        self.sess.close()
        # del self.sess, self.inputs, self.boxes
        tf.reset_default_graph()
        # tf.keras.backend.clear_session()
        super().close()


class Yolov3Main(Yolov3):
    def __init__(self, args):
        args.tiny = False
        super().__init__(args)


class Yolov3Tiny(Yolov3):
    def __init__(self, args):
        args.tiny = True
        super().__init__(args)
