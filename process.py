import cv2
import os
import sys
import numpy as np
from mrcnn import utils
import mrcnn.model as modellib
from Config import InferenceConfig


class PreProcessing:
    def __init__(self, root_dir, path_to_mask_rcnn):
        self.root_dir = root_dir
        self.path_to_mask_rcnn = path_to_mask_rcnn
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        self.COCO_MODEL_PATH = os.path.join(root_dir, "mask_rcnn_coco.h5")
        self._config = InferenceConfig()

    def import_mask_rcnn(self):
        # Import Mask RCNN
        sys.path.append(self.root_dir)  # To find local versifon of the library
        # Import COCO config
        sys.path.append(os.path.join(self.root_dir, "samples/coco/"))  # To find local version
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)

    def create_model_object(self):
        # Load weights trained on MS-COCO
        model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self._config)
        model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        self.background_subtractions(model)

    def background_subtractions(self, model):
        # Load the image - image will be the uploaded image passed into the function in production
         IMAGE_DIR = './test.jpg'
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
                mask_output[row:end_row + 1, col:end_col + 1] = r['masks'][row:end_row + 1, col:end_col + 1, s]

         cv2.imwrite('test_output.jpg', img_copy)
         cv2.imwrite('test_mask.jpg', (255*(mask_output.astype(np.uint8))))
         cv2.imwrite('test_segment.jpg', image * mask_output[..., None].astype(np.uint8))


path_to_mask_rcnn = '/home/travjav/Development/Mask_RCNN'
# Root directory of the Mask RCNN project
ROOT_DIR = os.path.abspath(path_to_mask_rcnn)
processingScript = PreProcessing(path_to_mask_rcnn, ROOT_DIR)
processingScript.import_mask_rcnn()
processingScript.create_model_object()
