"""
Module level docstring

"""
import os
import sys
import numpy as np
from skimage import transform
import pickle

def detect_sample_shape(sample):
    """Returns how many dimensions the sample is"""
    return len(sample.shape)

class Rescale():
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            #print('output_size is an int')
            self.output_shape = (output_size, output_size)
        elif isinstance(output_size, (tuple)):
            self.output_shape = output_size
        else:
            raise Exception(
                    ('outputs size given to Rescale Class',
                    ' is neither int nor tuple'))

    def __call__(self, sample):
        #print('Do we even call this function?')
        #print(f'input_shape: {sample.shape}')
        sample_dimensionality = detect_sample_shape(sample)
        if sample_dimensionality == 3:
            channels, height, width = sample.shape
            output_shape = (channels,
                            self.output_shape[0],
                            self.output_shape[1])
            resized_sample = transform.resize(image=sample,
                                              output_shape=output_shape,
                                              preserve_range=True)
        elif sample_dimensionality == 2:
            #print('Two dimensional case')
            resized_sample = transform.resize(image=sample,
                                              output_shape=self.output_shape,
                                              preserve_range=True
                                             )
            #print(f'resized_shape: {resized_sample.shape}')
        else:
            raise Exception('Sample shape is not recognized')
        return resized_sample

class Normalize():
    """Uses the mean pixel to normalize each image"""
    def __init__(self, path_to_mean_pixel):
        """Initializer method for Normalize callable"""
        with open(path_to_mean_pixel, 'rb') as fp:
            self.mean_pixel = pickle.load(fp)
        fp.close()

    def __call__(self,array):
        """subtracts the mean pixel from the array"""
        return array - self.mean_pixel
