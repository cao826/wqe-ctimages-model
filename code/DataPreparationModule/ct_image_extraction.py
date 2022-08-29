"""
Module level Docstring
"""
import os
from collections import namedtuple
from typing import Union, Any, List
import pydicom as dcm
import numpy as np
from pydicom.dataset import FileDataset
from pydicom.dicomdir import DicomDir
from PIL import Image
from matplotlib import cm

CtImages = namedtuple("CtImages", "no_window lung_window savefile_name")
lung_window = (1500, -600)

def read_in_dicom_file(
    dicom_path: str, dicom_filename: str
) -> Union[FileDataset, DicomDir]:
    """Reads a dicom file into the program memory"""
    return dcm.dcmread(os.path.join(dicom_path, dicom_filename))

def check_dicom_file(dicom_file: Union[FileDataset, DicomDir]) -> Any:
    """Checks there is nothing wrong with the dicom file that was read in"""
    spp_message = ("More than one sample per pixel",
            " (image is not grayscale)")
    photometric_interpretation_message = ("photometric ",
            "interpretation is not MONOCHROME2")
    assert dicom_file.SamplesPerPixel == 1, spp_message
    assert dicom_file.PhotometricInterpretation == "MONOCHROME2",\
                photometric_interpretation_message

def modality_lut(pixel_array, dicom_file):
    """Converts 'raw' pixel values to Hounsfield units

    The values in the pixel array data do not have any physical
    interpretation. In order to convert those values into physically
    meaningful measurements (of attenuation), we need to apply
    a 'modality lookup table', or modaluty LUT
    """
    rescale_slope = dicom_file.RescaleSlope
    rescale_intercept = dicom_file.RescaleIntercept
    hounsfield_array = pixel_array * rescale_slope + rescale_intercept
    return hounsfield_array

def apply_ct_window(img, window):
    """Windows the pixel data to produce a windowed CT image

    Windowing refers to changing two things:
    1. (WindowWidth) The range of hounsfield units that are represented by different shades of gray
    2. (WindowCenter) The midpoint of the range. Sometimes referred to as window level
    'Lung Window' refers to WindowWidth=1500, WindowCenter=-600 (approximately)

    the window argument is a tuple in the format: (WindowWidth, WindowCenter)
    """
    window_width, window_center = window
    windowed_image = (img - window_center + 0.5 * window_width) / window_width
    windowed_image[windowed_image < 0] = 0
    windowed_image[windowed_image > 1] = 1
    return windowed_image

def check_lung_window_image(lung_array):
    """Ensures the windowed image is valid"""
    assert (lung_array.min() >= 0.0), "lung window min not in [0, 1]"
    assert (lung_array.max() <= 1.0), "lung window mas not in [0, 1]"

def prepare_savefile_name(dicom_file):
    """Prepares a unique_filename to save a windowed image to file"""
    patient_id = dicom_file.PatientID
    study_date = dicom_file.StudyDate
    save_filename = "-".join([patient_id, study_date]) + ".png"
    return save_filename

def extract_raw_and_lung_window_arrays(path2slice, slice_filename):
    """Extracts a lung window CT image from a DICOM file

    Uses all of the functions above as helpers
    """
    dicom_file = read_in_dicom_file(
        dicom_path=path2slice, dicom_filename=slice_filename
    )
    check_dicom_file(dicom_file)
    raw_pixel_array = dicom_file.pixel_array
    hounsfield_array = modality_lut(pixel_array=raw_pixel_array, dicom_file=dicom_file)
    lung_array = apply_ct_window(img=hounsfield_array, window=lung_window)
    check_lung_window_image(lung_array)
    savefile_name = prepare_savefile_name(dicom_file)

    images_from_dicom = CtImages(hounsfield_array, lung_array, savefile_name)
    return images_from_dicom

def create_image_from_array(ct_array):
    """Converts an array from DICOM to a PIL Image instance"""
    ct_image = Image.fromarray(
        np.uint8(cm.gray(ct_array) * 255)
        ).convert('RGB')
    return ct_image

def extract_images(path2slice, slice_filename):
    """Extract a PIL Image instance of the lung window image"""
    dicom_arrays = extract_raw_and_lung_window_arrays(
            path2slice=path2slice,
            slice_filename=slice_filename
            )
    lung_window_image = create_image_from_array(dicom_arrays.lung_window)
    no_window_image = create_image_from_array(dicom_arrays.no_window)
    return lung_window_image, no_window_image, dicom_arrays.savefile_name

def save_image(image, path2savefolder, savefile_name):
    """saves the windowed image to file"""
    full_savepath = os.path.join(path2savefolder, savefile_name)
    image.save(full_savepath)
