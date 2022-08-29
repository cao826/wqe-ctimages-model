"""Simple code to find the mean pixel of the image dataset"""
import os
import pickle
import argparse
import numpy as np

def read_numpy_array(path, filename):
    """Reads in the numpy array"""
    filepath = os.path.join(path, filename)
    return np.load(file=filepath)

def get_sum_of(array):
    """Returns the sum of the array"""
    return np.sum(a=array)

def get_numpy_files(path):
    """returns a list of numpy files in the path"""
    files = os.listdir(path)
    return [file for file in files if '.npy' in file]

def save_result(savepath, result, month, date):
    """Saves the resultimg mean pixel to file"""
    savename = f'mean_pixel-{month}-{date}'
    destination = os.path.join(savepath, savename)
    with open(destination, 'wb') as file_object:
        pickle.dump(result, file_object)
    file_object.close()

def do_it_all(positives_path, negatives_path):
    """find the mean pixel of the dataset"""
    positive_filenames = get_numpy_files(path=positives_path)
    negative_filenames = get_numpy_files(path=negatives_path)

    positive_sums = [
            get_sum_of(read_numpy_array(positives_path, filename)) \
                    for filename in positive_filenames
            ]
    negative_sums = [
            get_sum_of(read_numpy_array(negatives_path, filename))\
                    for filename in negative_filenames
            ]
    total_sums = positive_sums + negative_sums
    return sum(total_sums) / (len(total_sums) * (512**2))

if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(
            description='Prepare the mean pixel and save it to file')
    my_parser.add_argument('-pospath')
    my_parser.add_argument('-negpath')
    my_parser.add_argument('-savepath')
    my_parser.add_argument('-month')
    my_parser.add_argument('-date')

    args = my_parser.parse_args()

    mean_pixel = do_it_all(positives_path=args.pospath,
                           negatives_path=args.negpath)
    save_result(savepath=args.savepath, result=mean_pixel,
                month=args.month, date=args.date)
