# How to run 
# python3 process.py <path to Mask RCNN project> <Path to image to test>

import cv2
import os
import sys
import numpy as np

path_to_mask_rcnn = sys.argv[1]

# Root directory of the Mask RCNN project
ROOT_DIR = os.path.abspath(path_to_mask_rcnn)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Load in an image
IMAGE_DIR = sys.argv[2]
image = cv2.imread(IMAGE_DIR)

# Run detection
results = model.detect([image], verbose=1)

# Visualize and save results
r = results[0]
img_copy = image.copy()
mask_output = np.zeros(img_copy.shape[:2], dtype=np.bool)
for s, (roi, class_id) in enumerate(zip(r['rois'], r['class_ids'])):
    if class_id == 1:
        row, col, end_row, end_col = roi
        cv2.rectangle(img_copy, (col, row), (end_col, end_row), (0, 0, 255))
        mask_output[row:end_row+1, col:end_col+1] = r['masks'][row:end_row+1, col:end_col+1,s]

cv2.imwrite('test_output.jpg', img_copy)
cv2.imwrite('test_mask.jpg', (255*(mask_output.astype(np.uint8))))
cv2.imwrite('test_segment.jpg', image * mask_output[...,None].astype(np.uint8))