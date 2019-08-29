import cv2
import os
import sys
import Config
import numpy as np
from mrcnn import utils
import mrcnn.model as modellib
from Config import InferenceConfig
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pickle

class Inference:
    def __init__(self, root_dir, path_to_image, path_to_model):
        self.root_dir = root_dir
        self.path_to_image = path_to_image
        self.path_to_model = path_to_model
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.MODEL_DIR = os.path.join(Config.ROOT_DIR, "logs")
        self.COCO_MODEL_PATH = os.path.join(root_dir, "mask_rcnn_coco.h5")
        self._config = InferenceConfig()

    def import_mask_rcnn(self):
        # Import Mask RCNN
        sys.path.append(self.root_dir)  # To find local versifon of the library
        # Import COCO config
        # To find local version
        sys.path.append(os.path.join(self.root_dir, "samples/coco/"))
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)

    def create_model_object_and_infer(self):
        # Load weights trained on MS-COCO
        model = modellib.MaskRCNN(
            mode="inference", model_dir=self.MODEL_DIR, config=self._config)
        model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        return self.background_subtractions(model)

    def background_subtractions(self, model):
        f = self.path_to_image
        print('File: ' + f)
        image = cv2.imread(f)

        # Run detection
        print('Run Mask RCNN and mask out image...')
        results = model.detect([image], verbose=1)
        # Get the results
        r = results[0]
        
        # Find all of the human detections and extract bounding boxes
        areas = []
        for s, (roi, class_id) in enumerate(zip(r['rois'], r['class_ids'])):
            if class_id == 1:
                row, col, end_row, end_col = roi
                areas.append((s, (end_col - col + 1) * (end_row - row + 1)))

        # Find the human bounding box with the largest area
        if len(areas) == 0:
            print('Warning: {} did not find any humans - quitting'.format(f))
            return

        max_val = max(areas, key=lambda x: x[1])

        # Get the bounding box coordinates
        row, col, end_row, end_col = r['rois'][max_val[0]]

        # Extract the mask and image - cropped
        mask_output = r['masks'][row:end_row + 1, col:end_col + 1, max_val[0]]
        crop = image[row:end_row + 1, col:end_col + 1]
        masked_image = crop * mask_output[..., None].astype(np.uint8)

        # Step #1 - Resize the image and normalize
        print('Reshape masked out image and normalize...')
        res = cv2.resize(masked_image, (511, 511))
        res = res.astype(np.float32) / 255

        # Step #2 - Load the model
        print('Load in model...')
        model = load_model(os.path.join(self.path_to_model, 'model_bodyfat.hdf5'))
        model._make_predict_function()

        # Extract the mapping from class ID to actual body fat label
        with open(os.path.join(self.path_to_model, 'labels.pkl'), 'rb') as f:
            dct = pickle.load(f)

        # Step #3 - Perform prediction on new image
        # Create a 1 x 511 x 511 x 3 array
        # The first dimension corresponds to the number
        # of examples we're feeding to the model, which is 1
        print('Make prediction...')
        pred = model.predict(res[None])[0]

        # Get the actual physical label for the body fat
        # Find the label with the highest probability
        ind = np.argmax(pred)
        return dct[ind], pred[ind]


path_to_image = '../BodyfatDataset/8_percent/4qkmro002-LMsYfgB.jpg'
path_to_model = './models'
processingScript = Inference(Config.ROOT_DIR, path_to_image, path_to_model)
processingScript.import_mask_rcnn()
label, pred = processingScript.create_model_object_and_infer()
print('Output label: {}'.format(label))
print('Probability: {}'.format(pred))

