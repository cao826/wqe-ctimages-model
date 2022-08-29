"""Module level docstring
"""
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

#fix random seed
torch.manual_seed(0)

def get_label(dataframe, filename):
    """
    Gets the value from the dataframe

    THIS FUNCTION ASSUMES:
    - there is a column named 'Filename' where the filname is
    - there is a column named 'Label' where the label is

    I could write a more general function, but it would be a pain
    """
    label = dataframe[dataframe.Filename == filename].Label.values[0]
    return label

def read_image(path2image):
    """Reads a stored array as a PyTorch tensor"""
    return Image.open(path2image)

def get_pid(filename):
    """returns the part of the filename that corresponds to pid
    """
    return filename.split('-')[0]

class NlstDataset(Dataset):
    """Dataset for Nlst images and clinical information
    """
    def __init__(self, path2clinical_data,
                 path2images, neg_subfolder,
                 pos_subfolder, transforms):
        """Constructor"""
        self.clinical_data = pd.read_csv(path2clinical_data)
        self.path2images = path2images
        self.subfolders = [neg_subfolder, pos_subfolder]
        self.label_to_subfolder = {
            1: pos_subfolder,
            0: neg_subfolder
        }
        self.pos_files = [file for file in os.listdir(os.path.join(
            path2images,
            pos_subfolder))
            if '.npy' in file]

        self.neg_files = [file for file in os.listdir(os.path.join(
            path2images,
            neg_subfolder))
            if '.npy' in file]

        self.files = self.neg_files + self.pos_files
        random.shuffle(self.files)
        print('dataset has {} files'.format(len(self.files)))
        self.transformations = transforms

    def __len__(self):
        """returns the number of datapoints"""
        return len(self.files)

    def get_label(self, filename):
        """gets the label of a scan slice"""
        label = 0
        if filename in self.pos_files:
            label = 1
        return label

    def get_clinical_info_vector(self, pid):
        """Returns the demographic information about a scan's patient"""
        mask = (self.clinical_data.pid == pid)
        pid_row = self.clinical_data[mask]
        assert pid_row.shape[0] == 1
        clinical_info_vec = pid_row[[
            'race',
            'cigsmok',
            'gender',
            'age'
        ]].values[0].astype(float)
        clinical_info_tensor = torch.tensor(clinical_info_vec)
        return clinical_info_tensor.float()

    def get_path_to_filename(self, filename, label):
        """returns the path to the filename"""
        subfolder = self.label_to_subfolder[label]
        intermediate = os.path.join(subfolder, filename)
        path2filename = os.path.join(self.path2images, intermediate)
        return path2filename


    def __getitem__(self, idx):
        """Returns a scan and its clinical information"""
        filename = self.files[idx]
        pid = get_pid(filename)
        label = self.get_label(filename)
        path2image = self.get_path_to_filename( filename, label)
        image = read_image(path2image)
        image = self.transformations(image)
        clinical_info = self.get_clinical_info_vector(int(pid))

        return image, clinical_info, label

if __name__ == 'main':
    print('Why are you running this a script? This is a module!')
