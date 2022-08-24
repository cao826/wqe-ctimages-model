"""
Module level docstring

"""
import pickle
import torch

class SubtractMeanPixel():
    """Uses the mean pixel to normalize each image"""
    def __init__(self, path_to_mean_pixel):
        """Initializer method for Normalize callable"""
        with open(path_to_mean_pixel, 'rb') as file_object:
            self.mean_pixel = torch.tensor(pickle.load(file_object))
        file_object.close()

    def __call__(self,array):
        """subtracts the mean pixel from the array"""
        return array - self.mean_pixel
