import sys, os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision
import datasets     # huggingface
from datasets import load_dataset, load_metric

from ai_helper import constants_dataset as c_d
from ai_helper import constants_ai_h as c
from ai_helper import dataset_load_helper as ds_l_h
from ai_helper import pandas_helper
from ai_helper import torch_help_functions

from helper import helpers as h
from helper import erik_functions_remote
from helper import erik_functions_files
from helper import constants_helper




# load one or all datasets from ctu13, None loads all rows
def dataset_load_ctu13(CTU13_nr=1, NR_ROWS=None):
    ctu13_dataset = []

    DIR_DATASET_CTU13 = c_d.DIR_DATASET_CTU13
    URL_CTU13 = c_d.URL_CTU13
    FILE_EXTENSION = constants_helper.FILE_EXTENSION_BINETFLOW

    # update dataset if necessarry
    #h.download_file(URL_CTU13, DATASET_BASEDIR)

    # load datset into a dataframe
    DIR_DATASET_CTU13_NRX = os.path.join(DIR_DATASET_CTU13, str(CTU13_nr))
    df = pandas_helper.dataframes_load(DIR_DATASET_CTU13_NRX, FILE_EXTENSION, NR_ROWS, c_d.CONCATENATE_TRUE)
    return df

def MNIST(dir_dataset, transforms_train, transforms_test, download = True):
    train_set = torchvision.datasets.MNIST(root=dir_dataset, train=True, transform=transforms_train, download=True)
    test_set = torchvision.datasets.MNIST(root=dir_dataset, train=False, transform=transforms_test, download=True)
    return train_set, test_set


def FashionMNIST(dir_dataset, transforms_train, transforms_test, download = True):
    train_set = torchvision.datasets.FashionMNIST(root=dir_dataset, train=True, transform=transforms_train, download=True)
    test_set = torchvision.datasets.FashionMNIST(root=dir_dataset, train=False, transform=transforms_test, download=True)
    return train_set, test_set

def Cifar10(p, transform_train, transform_val, tranform_test = None):
    train_ds = torchvision.datasets.CIFAR10(c.DIR_DATASETS, train=True, download=True, transform=transform_train)
    val_ds = torchvision.datasets.CIFAR10(c.DIR_DATASETS, train=False, download=True, transform=transform_val)
    test_ds = None

    val_size = 0
    train_size = len(train_ds) - val_size

    print('train size : ' + str(len(train_ds)))
    print('img_size : ' + str(p[c.AUGMENT_CROP]))

    train_ds, _ = torch_help_functions.random_split(train_ds, [train_size, val_size])
    # test_ds = torchvision.datasets.CIFAR10(c.DIR_DATASETS, train=False, download=True, transform=transform_test)
    # val_ds = torchvision.datasets.CIFAR10(c.DIR_DATASETS, train=False, download=True, transform=transform_test)

    return train_ds, val_ds, test_ds


def eurosat_rgb(transformations=None):      # ToDo add transforms as a paramter
    print('Loading dataset : EuroSat RGB ')

    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]

    if transformations == None:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std)
        ])

    ds_all = ImageFolder(root=c_d.DIR_DATASET_EUROSAT_RGB, transform=transformations)
    datasets = ds_l_h.train_val_dataset(ds_all, 0.2)
    datasets_val_test = ds_l_h.train_val_dataset(datasets['val'], 0.5)

    ds_train = datasets['train']
    ds_val = datasets_val_test['train']
    ds_test = datasets_val_test['val']

    return ds_train, ds_val, ds_test

def eurosat_ms(transformations=False):      # ToDo add transforms as a paramter
    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]

    if not transformations:
        transformations = torchvision.transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std)
        ])

    ds_all = ImageFolder(transform=transformations, root=c_d.DIR_DATASET_EUROSAT_MS)
    datasets = ds_l_h.train_val_dataset(ds_all, 0.2)
    datasets_val_test = ds_l_h.train_val_dataset(datasets['val'], 0.5)

    ds_train = datasets['train']
    ds_val = datasets_val_test['train']
    ds_test = datasets_val_test['val']

    return ds_train, ds_val, ds_test


#   download weights to local to use in realesr and gfpgans
def download_weights_realesr_gan(urls=c_d.URLS_WEIGHTS_GFP_GAN, path_save=c_d.DIR_MODEL_WEIGHTS_GFPGAN):
    is_successful = True
    erik_functions_files.make_dir(path_save)
    for url in urls:
        basename = os.path.basename(url)
        path_file = os.path.join(path_save, basename)
        if not(erik_functions_files.file_exists(path_file, supress_not_found=False)):
            print('Downloading ' + url)
            success = erik_functions_remote.wget_download(url, path_save)
            if not success: is_successful = False
    return is_successful



# ToDo save a preprocessed dataset to disk in a good load format
def dataset_save_to_disk(dataset, path_save, format_save):
    pass


# load image dataset from imagefolders
def dataset_load_from_imagefolder(dir_dataset_base, dataset_small=False):
    #   def dataset_dirs_from_base(dataset_dir_base, dir_split=c_d.DATASET_SPLIT, dir_small=c_d.DATASET_SMALL, small=False):
    train_dir_split, val_dir_split, test_dir_split = ds_l_h.dataset_dirs_from_base(dir_dataset_base, dir_small=dataset_small, small=dataset_small)
    print(train_dir_split)

    train_paths = ds_l_h.flatten_dirs_to_list(train_dir_split)
    val_paths = ds_l_h.flatten_dirs_to_list(val_dir_split)
    test_paths = ds_l_h.flatten_dirs_to_list(test_dir_split)

    dataset = load_dataset("imagefolder", data_files={"train": train_paths, "test": test_paths, "val": val_paths}, cache_dir=c_d.DIR_DATASET_CACHE)
    return dataset



if __name__ == "__main__":
    ctu13_1 = dataset_load_ctu13(1)

    print(ctu13_1.head())
    print(ctu13_1.info())
    print(ctu13_1.corr())

    LABEL = 'target'
    LABEL_FROM = 'Label'
    SUBSTRING = 'botnet'

    pandas_helper.df_column_create_contains_text(ctu13_1, LABEL, LABEL_FROM, SUBSTRING)

    print(ctu13_1.head())

    pandas_helper.pandas_dataframe_describe(ctu13_1)