import cv2
import os
import sys
import Config
import numpy as np
from mrcnn import utils
import mrcnn.model as modellib
from Config import InferenceConfig
import glob
from pathlib import Path


class PreProcessing:
    def __init__(self, root_dir, base_image_dir):
        self.root_dir = root_dir
        self.base_image_dir = base_image_dir
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.MODEL_DIR = os.path.join(Config.ROOT_DIR, "logs")
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
        # Iterate through all files with extension JPG
        files = list(Path(self.base_image_dir).glob('**/*.jpg'))
        for l, file in enumerate(files):
            # Access the file name and read in the image
            f = str(file)
            print('File: ' + f + " - {} / {}".format(l + 1, len(files)))
            image = cv2.imread(f)

            # Run detection
            results = model.detect([image], verbose=1)
            # Visualize and save results
            r = results[0]
            #img_copy = image.copy()
            mask_output = np.zeros(image.shape[:2], dtype=np.bool)

            # Find all of the human detections and extract bounding boxes
            areas = []
            for s, (roi, class_id) in enumerate(zip(r['rois'], r['class_ids'])):
                if class_id == 1:
                   row, col, end_row, end_col = roi
                   areas.append((s, (end_col - col + 1) * (end_row - row + 1)))

            # Find the human bounding box with the largest area
            if len(areas) == 0:
                print('Warning: {} did not find any humans - skipping'.format(f))
                continue

            max_val = max(areas, key=lambda x: x[1])

            # Get the bounding box coordinates
            row, col, end_row, end_col = r['rois'][max_val[0]]

            # Draw the rectangle
            #cv2.rectangle(img_copy, (col, row), (end_col, end_row), (0, 0, 255))

            # Extract the mask
            mask_output[row:end_row + 1, col:end_col + 1] = r['masks'][row:end_row + 1, col:end_col + 1, max_val[0]]

            # Create directory for the images for this label
            try:
                subdir = file.relative_to(*file.parts[:1])
                to_write_file = os.path.join(self.output_dir, str(subdir))
                to_write, _ = os.path.split(to_write_file)
                os.makedirs(to_write)
            except OSError:
                pass

            try:
                # Create mask subdirectory
                d = os.path.join(to_write, 'mask')
                os.makedirs(d)
            except OSError:
                pass

            try:
                # Create segmented subdirectory
                d = os.path.join(to_write, 'segment')
                os.makedirs(d)
            except OSError:
                pass

            filename_processed = self.create_segmented_filename()
            #cv2.imwrite('./sorted_8_percent/output' + filename_processed, img_copy)
            out = os.path.join(to_write, 'mask', filename_processed)
            print('Writing mask to: ' + out)
            cv2.imwrite(out, (255*(mask_output.astype(np.uint8))))
            out = os.path.join(to_write, 'segment', filename_processed)
            print('Writing segmented image to: ' + out)
            cv2.imwrite(out, image * mask_output[..., None].astype(np.uint8))

    def create_segmented_filename(self):
        import uuid
        return str(uuid.uuid4()) + '.jpg'

    def create_processed_images_directory(self, dir_):
        try:
            self.output_dir = dir_
            os.mkdir(dir_)
        except OSError:
            print("Creation of the directory failed")
        else:
            print("Successfully created the directory")


base_image_dir = '../BodyfatDataset'
output_dir = './output'
processingScript = PreProcessing(Config.ROOT_DIR, base_image_dir)
processingScript.create_processed_images_directory(output_dir)
processingScript.import_mask_rcnn()
processingScript.create_model_object()
