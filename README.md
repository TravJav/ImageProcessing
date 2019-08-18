# Setup (WIP)

1. Clone Mask RCNN project: https://github.com/matterport/Mask_RCNN
2. Install their requirements in their repo.  They're also here in this repo:
    `pip install -r requirements.txt`
3. The steps to go from preparing the data to training the model are as follows:

    * In `process.py`, change the `base_image_dir` and `output_dir` variables at
      the bottom of the script to be where the body fat images are (i.e. the location
      of where the Body Fat repo is: https://github.com/TravJav/BodyFatDataset and
      the desired output directory where the output images will be saved respectively.
    * Run the `process.py` script which will run Mask RCNN on each of the images
      and saved the cropped and masked human images and their corresponding masks:
     
          $ python3 process.py

      This will save the segmented images and masks in the directory specified in
      `output_dir`
    * Next in the `decompose_dataset.py` script, change the `base_dir` and the
      `output_dir` to be where the output images from the `process.py` file are
      where you'd like the training and validation images to be saved.
    * Run the `decompose_dataset.py` file which will split up the images into
      training and validation test datasets:

          $ python3 decompose_dataset.py
    * Run the training script which will create a deep learning model to learn
      how to infer the body fat of a human when their upper midsection is showing.

          $ python3 train.py
    * Reloading the trained model and performing inference is forthcoming.
