"""
Module level docstring
"""
import os
import pickle
from sklearn import metrics

def check_day(day):
    """Ensures the day is a day of the month"""
    if day > 31 or day < 0:
        raise Exception('Day specified is invalid')

def add_number(path2files, savefile, filetype, number=0):
    """
    Assumes the savepath is already verified
    """
    full_savefile = savefile + str(number) + filetype
    candidate_savepath = os.path.join(path2files, full_savefile)
    if not os.path.isfile(candidate_savepath):
        return full_savefile
    return add_number(path2files=path2files,
                      savefile=savefile,
                      filetype=filetype,
                      number=number+1)

class DateMaker():
    """creates dates in format for naming convention"""
    def __init__(self):
        self.month_mapper = {
            'August': 'aug',
            'September': 'sept',
            'October': 'oct',
            'November': 'nov'
            }
    def check_month(self, month):
        """check if month is recognized"""
        if month not in self.month_mapper:
            raise Exception('Month given is not an approved month')

    def check_date(self, month, day):
        """Checks the month and day"""
        self.check_month(month),
        check_day(day)

    def print_months(self):
        """Show which months are listed in the file naming scheme"""
        print(self.month_mapper)

    def __call__(self, month, day):
        """Creates date in naming convention"""
        self.check_date(month, day)
        return str(day) + f'_{month}_2022-'


class ResultMaker():
    """Returns proper return type"""

    def __init__(self):
        """
        """
        self.result_type_mapper = {
            'roc plot': 'roc_plot',
            'roc curve': 'roc_curve',
            'auroc': 'auroc',
            'accuracy': 'acc',
            'confusion matrix': 'conf_mat',
            'tsne plot': 'tsne_plot',
            'heatmap': 'activation_heatmap',
            'model weights': 'model_weights'
            }
    def check_result_type(self, result_type):
        """check to see if result type is recognized"""
        if result_type not in self.result_type_mapper:
            raise Exception('Invalid result type')

    def __call__(self, result_type):
        """Contributes the result type to the naming convention"""
        self.check_result_type(result_type)
        return self.result_type_mapper[result_type] + '-'

class FileExtension(): ##you should be able to infer filetype from the result_type
    """Adds file extension to naming convention"""

    def __init__(self):
        self.filetype_mapper = {
            'png': '.png',
            'pickle': '.pickle',
            'numpy': '.npy'
            'pytorch': '.pt'# HAS TO BE EXTENDED LATER
        }
    def check_filetype(self, filetype):
        """makes sure that the filetype is recognized"""
        if not filetype in self.filetype_mapper:
            raise Exception('uknown filetype')

    def __call__(self, filetype):
        """Returns proper filetype"""
        self.check_filetype(filetype)
        return self.filetype_mapper[filetype]

def make_savepath(month, day, result_type, filetype, path2file):
    """Makes full savepath"""
    date = DateMaker()(month, day)
    result_type_str = ResultMaker()(result_type)
    filetype = FileExtension()(filetype)
    savefile = date + result_type_str
    savefile = add_number(path2file, savefile, filetype)
    savepath = os.path.join(path2file, savefile)
    return savepath

#This is the code for actually saving to file

def check_for_overwrite(savepath):
    """checks if there is already a file with the savepath given"""
    if os.path.isfile(savepath):
        raise Exception(("file specified in savepath already exists. ",
                         "Do you want to overwrite?"))

def convert_to_pickle_file(thing, savepath):
    """
    THIS FUNCTION ASSUMES YOU HAVE ALREADY CHECKED FOR OVERWRITE!
    DO NOT USE THIS FUNCTION WILLY-NILLY!
    """
    with open(file=savepath, mode='wb') as fp:
        pickle.dump(thing, fp)
    fp.close()

def save_image_to_file(fig, savepath):
    """
    Will save a matplotlib figure object to file as a png. Will make sure that
    the savepath does NOT exist before saving and will throw a fit if you
    try to overwrite
    """
    check_for_overwrite(savepath)
    fig.savefig(savepath)

def save_accuracy(acc, savepath):
    """
    Saves the accuracy. I am not sure why this function needs to be
    explicitly typed like this, but whatever.
    """
    check_for_overwrite(savepath)
    convert_to_pickle_file(acc, savepath)

def save_roc_plot(y_test, y_pred_proba, savepath):
    """Saves the plot of the receiver-operator curve"""
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    check_for_overwrite(savepath)
    convert_to_pickle_file((fpr, tpr, _), savepath)
