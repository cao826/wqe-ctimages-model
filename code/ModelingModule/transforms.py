"""
Module level docstring

"""
import os
import sys
import numpy as np
from skimage import transform


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
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, (tuple)):
            self.output_size = output_size
        else:
            raise Exception('outputs size given to Rescale Class is neither int nor tuple')
        #self.output_size = output_size

    def __call__(self, sample):
        channels, height, width = sample.shape
        #print('{} channels, height {}, and width {}'.format(channels, height, width))
        output_shape = (channels, self.output_size[0], self.output_size[1])
        resized_sample = transform.resize(sample, output_shape, preserve_range=True)
        return resized_sample
