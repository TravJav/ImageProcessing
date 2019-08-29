# Setup (WIP)

1. Clone Mask RCNN project: https://github.com/matterport/Mask_RCNN
2. Install their requirements in their repo.  They're also here in this repo:
    `pip install -r requirements.txt`
3. The steps to go from preparing the data to training the model are as follows:

    * In `Config.py`, change the `path_to_mask_rcnn` variable to where the Mask
      RCNN project was cloned to
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
      Change `train_segment_dir` and `val_segment_dir` to be where the training
      and validation images are stored after running `decompose_dataset.py` at
      the bottom of the script.  You can also change the batch size, image size
      number of epochs and learning rate as well.  The `find_lr` mode will help
      determine what the most optimal learning rate is by increasing it after
      each epoch up to the maximum.  The losses after each epoch are saved
      to `results.csv` which you can examine for the optional learning rate.
      You can then run the `train` mode with this optimal learning rate to
      perform final training.

          $ python3 train.py
    * To perform inference, examine the `predict.py` file and modify
      the `path_to_image` and `path_to_model` to be the path used for
      the image to perform a prediction on and the directory of where
      the model got saved (this should be in the `models` subdirectory).
      The inference will run Mask RCNN on the image, crop and mask
      the image, resize it so that it's compatible for the model and
      determine the label for the image.  You can run this by:
      
          $ python3 predict.py
