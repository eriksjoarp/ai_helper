import torchvision 

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

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
