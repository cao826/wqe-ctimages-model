"""Module level docstring
"""
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

#fix random seed
torch.manual_seed(0)

def read_image(path2image):
    """Reads a stored array as a PyTorch tensor"""
    return Image.open(path2image)

def get_pngs(path: str):
    """gets the pngs files"""
    pngs = [filename for filename in os.listdir(path) if '.png' in filename]
    return pngs

def get_pid(filename):
    """returns the part of the filename that corresponds to pid
    """
    return filename.split('-')[0]

def check_clinical_data(table):
    """cod"""
    columns = table.columns
    assert "Image Index" in columns
    assert "Finding Labels" in columns
    assert "Patient Gender" in columns
    assert "Patient Age" in columns

class Chex8Dataset(Dataset):
    """Dataset for Nlst images and clinical information
    """
    def __init__(self, path2clinicaldata,
                 path2images, transforms):
        """Constructor"""
        self.clinical_data = pd.read_csv(path2clinicaldata)
        check_clinical_data(self.clinical_data)
        self.path2images = path2images
        self.files = get_pngs(self.path2images)
        print(f'dataset has {len(self.files)} files')
        self.transformations = transforms

    def __len__(self):
        """returns the number of datapoints"""
        return len(self.files)

    def get_label(self, filename):
        """gets the label of a scan slice"""
        row = self.clinical_data[self.clinical_data["Image Index"]]
        if row.shape[0] != 1:
            raise Exception('Something went wrong')
        label = row["Finding Labels"].values[0]
        return label

    def get_clinical_info(self, filename):
        """Returns the demographic information about a scan's patient"""
        mask = (self.clinical_data['Image Index'] == filename)
        patient_row = self.clinical_data[mask]
        if patient_row.shape[0] != 1:
            raise Exception('there is not exactly one match')
        patient_age = patient_row["Patient Age"].values[0]
        patient_gender = patient_row["Patient Gender"].values[0]
        if patient_gender == 'M':
            patient_gender = 1
        else:
            patient_gender = 0
        clinical_info_tensor = torch.tensor([patient_age, patient_gender]).float()
        return clinical_info_tensor

    def __getitem__(self, idx):
        """Returns a scan and its clinical information"""
        filename = self.files[idx]
        clinical_info_tensor = self.get_clinical_info(filename)
        label = self.get_label(filename)
        image = read_image(os.path.join(self.path2images, filename))
        image = self.transforms(image)
        return image, clinical_info, label

if __name__ == 'main':
    print('Why are you running this a script? This is a module!')
