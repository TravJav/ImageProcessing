from pathlib import Path
import random
import os
import shutil

base_dir = './output/BodyfatDataset/'
output_dir = './output_seg'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
files = list(Path(base_dir).glob('**/*.jpg'))

# Split up the data into training and validation
split_fraction = 0.8
random.shuffle(files)
split_point = int(len(files)*split_fraction)

# Gets the relevant file names
train_files = files[:split_point]
val_files = files[split_point:]

# Make the training and validation directories
try:
    os.makedirs(train_dir)
except OSError:
    pass

try:
    os.makedirs(val_dir)
except OSError:
    pass

# For each file in the training files...
for file in train_files:

    # If it's a mask, don't worry about it
    s = str(file)
    print('Training image: {}'.format(s))
    if 'mask' in s:
        continue

    # This should be the segmented image
    # Remove the segment part of the path
    s = s.replace('/segment', '/')
    s = s.replace(base_dir, '')
    f = Path(s)

    # Specify where we're going to copy to
    subdir = f.relative_to(*f.parts[:1])
    to_write_file = os.path.join(output_dir, 'train', str(subdir))
    to_write, _ = os.path.split(to_write_file)
    print('Copying to: {}'.format(to_write_file))
    # Make sure the directory exists
    try:
        os.makedirs(to_write)
    except OSError:
        pass

    # Copy the file over
    shutil.copyfile(str(file), str(to_write_file))

# Repeat for validation
for file in val_files:
    s = str(file)
    print('Training image: {}'.format(s))
    if 'mask' in s:
        continue

    s = s.replace('/segment', '/')
    s = s.replace(base_dir, '')
    f = Path(s)
    subdir = f.relative_to(*f.parts[:1])
    to_write_file = os.path.join(output_dir, 'val',  str(subdir))
    to_write, _ = os.path.split(to_write_file)
    print('Copying to: {}'.format(to_write_file))
    try:
        os.makedirs(to_write)
    except OSError:
        pass
    shutil.copyfile(str(file), str(to_write_file))
