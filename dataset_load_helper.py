import os.path
import random
import sys
import torchvision
import torch
#from fastai.vision.all import *
import PIL

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from helper import erik_functions_files
from helper import erik_functions_help_files_high
from ai_helper import constants_dataset as c_d


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets[c_d.DATASET_TRAIN] = Subset(dataset, train_idx)
    datasets[c_d.DATASET_VAL] = Subset(dataset, val_idx)
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


def dataset_directory_small(dir_dataset, dir_extension=c_d.DATASET_SMALL):
    dir_parent = erik_functions_files.dir_parent(dir_dataset)
    dir_base = os.path.basename(dir_dataset)
    dir_dataset_small = os.path.join(dir_parent, dir_base + dir_extension)
    bExists = erik_functions_files.path_exists(dir_dataset_small)
    return dir_dataset_small, bExists


# create a small subsection from a large dataset
def dataset_create_small(dir_dataset_files, nr_files_per_dir=1000, randomize_files=True, remove_old_dir_to=True, dir_extension=c_d.DATASET_SMALL):
    dir_parent = erik_functions_files.dir_parent(dir_dataset_files)
    dir_base = os.path.basename(dir_dataset_files)
    dir_dataset_files_out = os.path.join(dir_parent, dir_base + dir_extension)

    #def dirs_truncate_to_out_dir(dir_from, dir_to, nr_files_per_dir = 100, randomize_files = True, remove_old_dir_to=True):
    erik_functions_help_files_high.dirs_truncate_to_out_dir(dir_dataset_files, dir_dataset_files_out, nr_files_per_dir=nr_files_per_dir,
                                                            randomize_files=randomize_files, remove_old_dir_to=remove_old_dir_to)

# split a list into train, val, test portions
def dataset_split(paths, val_split=0.1, test_split=0.1):
    nr_total = len(paths)
    random.shuffle(paths)
    val_crossover = int((1-val_split-test_split)*nr_total)
    test_crossover = int((1 - test_split) * nr_total)

    paths_train = paths[:val_crossover]
    paths_val = paths[val_crossover:test_crossover]
    paths_test = paths[test_crossover:]

    return paths_train, paths_val, paths_test

# create train,val,test dirs from a dataset folder with the labels in directorys. split them accordingly
def dataset_create_train_test_val(dir_dataset_files, val_split=0.1, test_split=0.1, delete_old=True):
    dir_split, _ = dataset_directory_small(dir_dataset_files, dir_extension=c_d.DATASET_SPLIT)

    dir_train = os.path.join(dir_split, c_d.DATASET_TRAIN)
    dir_val = os.path.join(dir_split, c_d.DATASET_VAL)
    dir_test = os.path.join(dir_split, c_d.DATASET_TEST)

    # get all path_files into lists
    sub_dirs = erik_functions_files.dirs_in_dir(dir_dataset_files, full_path=True)
    print(sub_dirs)

    # create train, test, val dir
    for sub_dir in sub_dirs:
        base_dir = os.path.basename(sub_dir)
        sub_dir_files = erik_functions_files.files_in_dir_full_path(sub_dir)

        train_paths, val_paths, test_paths = dataset_split(sub_dir_files, val_split, test_split)

        dir_train_sub = os.path.join(dir_train, base_dir)
        dir_val_sub = os.path.join(dir_val, base_dir)
        dir_test_sub = os.path.join(dir_test, base_dir)

        print(dir_train_sub)

        erik_functions_files.copy_files(train_paths, dir_train_sub, delete_old=delete_old)
        erik_functions_files.copy_files(val_paths, dir_val_sub, delete_old=delete_old)
        erik_functions_files.copy_files(test_paths, dir_test_sub, delete_old=delete_old)


# create directory names from a basedir using train,val,test
def dataset_dirs_split(dataset_path, split=c_d.DATASET_SPLIT):
    dir_split, _ = dataset_directory_small(dataset_path, dir_extension=split)

    dir_train = os.path.join(dir_split, c_d.DATASET_TRAIN)
    dir_val = os.path.join(dir_split, c_d.DATASET_VAL)
    dir_test = os.path.join(dir_split, c_d.DATASET_TEST)

    return dir_train, dir_val, dir_test


def dataset_dirs_from_base(dataset_dir_base, dir_split=c_d.DATASET_SPLIT, dir_small=c_d.DATASET_SMALL, small=False):
    if small:
        dir_train, dir_val, dir_test = dataset_dirs_split(dataset_dir_base, dir_split + dir_small)
    else:
        dir_train, dir_val, dir_test = dataset_dirs_split(dataset_dir_base, dir_split)
    return dir_train, dir_val, dir_test




if __name__=='__main__':
    dir_dataset = r'C:\ai\datasets\eurosat\EuroSAT\2750_32'
    dir_dataset, _ = dataset_directory_small(dir_dataset)

    print(dir_dataset)
    dataset_create_train_test_val(dir_dataset)

    #dataset_create_small(dir_dataset, nr_files_per_dir=1000, randomize_files=True, remove_old_dir_to=True)
    #dataset_create_train_test_val(dir_dataset)

    #dir_dataset = r'C:\ai\datasets\eurosat\EuroSAT\2750_16'
    #dataset_create_train_test_val(dir_dataset)







