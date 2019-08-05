# Setup (WIP)

1. Clone Mask RCNN project: https://github.com/matterport/Mask_RCNN
2. Install their requirements in their repo.  They're also here in this repo:
    `pip install -r requirements.txt`
3. To run the person segmentation test script, run like the following:
    `python3 process.py <path to Mask RCNN project> <path to image>`
    
    Three images will be saved to disk:
    * The original image with the people delineated in bounding boxes
    * A binary mask that isolates out where the people are
    * The background subtracted image containing just the people from the original image