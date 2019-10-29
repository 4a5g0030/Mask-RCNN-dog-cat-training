import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import pycocotools.coco
import mrcnn.utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config


ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
MODEL_WEIGHT_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_animal_0005.h5')

if not os.path.exists(MODEL_WEIGHT_PATH):
    print('Did not find this weight' + MODEL_WEIGHT_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, 'dataset/pic')


class BottleConfig(Config):
    NAME = "animal"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 2


class InferenceConfig(BottleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_WEIGHT_PATH, by_name=True)
class_names = ['BG', 'dog', 'cat']
file_names = next(os.walk(IMAGE_DIR))[2]


image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
results = model.detect([image], verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])