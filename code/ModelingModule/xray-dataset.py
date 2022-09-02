"""Module level docstring
"""
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

def read_image(filename, path):
    """returns a PIL Image instance"""
    path2image = os.path.join(path, filename)
    return PIL.Image.open(path2image)

def check_table(table: pd.DataFrame):
    """checks that it IS the clinical data"""
    columns = table.columns
    assert 'Image Index' in columns
    assert 'Finding Labels' in columns
    assert 'Patient ID' in columns
    assert 'Patient Age' in columns
    assert 'Patient Gender' in columns

class XrayDataset(Dataset):
    """Dataset class for cxr8 dataset"""
    def __init__(self, transforms, path2images, path2clinicaldata):
        """Constructor"""
        super().__init__()
        self.transforms  = transforms
        self.path2images = path2images
        self.files = [
                filename in os.listdir(self.path2images) if '.png' in filename
                ]
        self.clinical_data = pd.read_csv(pathclinicaldata)
        check_table(self.clinical_data)

    def get_label(self, filename):
        """returns the label of the filename"""
        filename_row = self.clinical_data[self.clinical_data['Image Index'] == filename]
        assert filename_row.shape[0] == 1, 'more than one file row found'
        label

    def __len__(self):
        """returns the length of the dataset
        otherwise known as the number of examples
        """
        return len(self.files)

    def __getitem__(self, idx):
        """returns a single image and its label"""
        filename = self.files[idx]
        image = read_image(self.path2images, filename)



