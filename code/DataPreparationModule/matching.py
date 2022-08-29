"""
Module Level Docstring
"""

import os
import sys
import shutil
import pandas as pd

def get_path_to_pid_dicoms(path2pid):
    """Finds where in the PID folder the dicoms are."""
    lst_of_nonempty_dirs = []
    for root, dirs, files in os.walk(path2pid):
        if not files:
            continue
        else:
            lst_of_nonempty_dirs.append(root)
    assert lst_of_nonempty_dirs, 'More than one non-empyt subdirectory'
    return lst_of_nonempty_dirs[0]

def get_dicom_files_for(path2pid):
    """Returns list of dicom files for a PID path"""
    path_2_dicoms = get_path_to_pid_dicoms(path2pid)
    return [file for file in os.listdir(path_2_dicoms) if not ('._' in file) ]

def get_year_from(year_foldername: str):
    """
    """
    return int(year_foldername.split('-')[2])


class PidHandler():
    """
    """

    def __init__(self, root, pid):
        """
        """
        self.pid = pid
        self.root = root
        self.path_to_dicoms = get_path_pid_dicoms(self.root)
        self.dicoms = get_dicom_files_for(self.root)
        self.get_year_dict()

    def get_year_dict(self):
        """
        """
        year_directories = [
                folder for folder in os.listdir(self.root) if folder[0] != '.'
                ]
        assert len(year_directories) > 0, ('No year folders detected for',
                f'pid: {pid}'))
        self.year_dict = {
                get_year_from(year_foldername): year_foldername for year_foldername in year_directories
                }

    def get_slice(slice_num):
        """
        """

