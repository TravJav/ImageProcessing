from pathlib import Path
import random
import os
import shutil

class DecomposeDataset(object):
    def __init__(self, base_dir, output_dir, split_fraction):
        self.output_dir = output_dir
        self.base_dir = base_dir
        self.train_dir_segment = os.path.join(output_dir, 'train_segment')
        self.train_dir_mask = os.path.join(output_dir, 'train_mask')
        self.val_dir_segment = os.path.join(output_dir, 'val_segment')
        self.val_dir_mask = os.path.join(output_dir, 'val_mask')

        self.files = list(Path(base_dir).glob('**/*.jpg'))
        self.split_fraction = split_fraction

    def split(self):
        # Split up the data into training and validation
        ind = list(range(len(self.files)))
        random.shuffle(ind)
        split_point = int(len(self.files)*self.split_fraction)

        # Gets the relevant file names
        train_files = [self.files[i] for i in ind[:split_point]]
        val_files = [self.files[i] for i in ind[split_point:]]

        # Make the training and validation directories
        try:
            os.makedirs(self.train_dir_segment)
        except OSError:
            pass

        try:
            os.makedirs(self.train_dir_mask)
        except OSError:
            pass

        try:
            os.makedirs(self.val_dir_segment)
        except OSError:
            pass

        try:
            os.makedirs(self.val_dir_mask)
        except OSError:
            pass

        # For each file in the training files...
        for file in train_files:
            s = str(file)

            # Specify where we're going to copy to
            subdir = os.path.relpath(s, self.base_dir)
            to_write_segment = os.path.join(self.train_dir_segment, subdir)
            to_write_mask = to_write_segment.replace('train_segment', 'train_mask')
            file_mask = s.replace('segment', 'mask')
            to_write1, _ = os.path.split(to_write_segment)
            to_write2, _ = os.path.split(to_write_mask)
            # Make sure the directories exists
            try:
                os.makedirs(to_write1)
            except OSError:
                pass

            try:
                os.makedirs(to_write2)
            except OSError:
                pass

            # Copy the files over
            shutil.copyfile(s, to_write_segment)
            shutil.copyfile(file_mask, to_write_mask)

        # Repeat for validation
        for file in val_files:
            s = str(file)

            # Specify where we're going to copy to
            subdir = os.path.relpath(s, self.base_dir)
            to_write_segment = os.path.join(self.val_dir_segment, subdir)
            to_write_mask = to_write_segment.replace('val_segment', 'val_mask')
            file_mask = s.replace('segment', 'mask')
            to_write1, _ = os.path.split(to_write_segment)
            to_write2, _ = os.path.split(to_write_mask)
            # Make sure the directories exists
            try:
                os.makedirs(to_write1)
            except OSError:
                pass

            try:
                os.makedirs(to_write2)
            except OSError:
                pass

            # Copy the file over
            shutil.copyfile(s, to_write_segment)
            shutil.copyfile(file_mask, to_write_mask)

base_dir = './output/segment'
output_dir = './output_seg'
split_fraction = 0.8
obj = DecomposeDataset(base_dir, output_dir, split_fraction)
obj.split()
