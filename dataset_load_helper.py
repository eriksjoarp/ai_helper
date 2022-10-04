import os.path
import sys
import torchvision
import torch
from fastai.vision.data import ImageList

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

############################        IMPORT HELPER MODULES       ############################
sys.path.append(os.getcwd() + '/..')
import python_imports
for path in python_imports.dirs_to_import(): sys.path.insert(1, path)
############################################################################################

import constants_dataset as c_d
import erik_functions_files
import PIL

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def torchvision_datasets(print_names = False):
    dataset_names = dir(torchvision.datasets)
    dataset_names_return = []
    for dataset_name in dataset_names:
        if dataset_name[0].isupper():
            dataset_names_return.append(dataset_name)
            if print_names: print(dataset_name)
    return dataset_names_return

def label_to_id(path_labels):
    # load list of labels
    filename = os.path.basename(path_labels)
    directory = os.path.dirname(path_labels)

    labels = erik_functions_files.get_filelines_to_list(directory, filename)

    # create mapping of labels
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label


def openMultiChannelImage(fpArr):
    '''
    Open multiple images and return a single multi channel image
    '''
    mat = None
    nChannels = len(fpArr)
    for i, fp in enumerate(fpArr):
        img = PIL.Image.open(fp)
        chan = PIL.pil2tensor(img, dtype='float').div_(255)
        if (mat is None):
            mat = torch.zeros((nChannels, chan.shape[1], chan.shape[2]))
        mat[i, :, :] = chan
    return PIL.Image(mat)


#class MultiChannelImageItemList(ImageItemList):
class MultiChannelImageItemList(ImageList):
    def open(self, fn):
        return openMultiChannelImage(fn)







