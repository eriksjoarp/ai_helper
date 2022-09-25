import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import time
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import constants_ai_h as c



######################              OPTIMISERS              #######################




######################              TRANSFORMS              #######################

def dataset_transforms(p):
    # PARAMETER
    trans_normalize = transform_normalize_cifar10()

    rand_trans = [transforms.RandomRotation(30), transforms.RandomHorizontalFlip(), transforms.RandomPerspective(), transforms.RandomAffine(20), transforms.RandomInvert()]

    # preparing datasets
    # PARAMETER
    if p[c.AUGMENT] == c.AUGMENT_RANDAUGMENT:
        #transform_train = transforms.Compose([transforms.Resize((p[c.AUGMENT_RESIZE]), antialias=True), transforms.RandomApply(rand_trans), transforms.ToTensor(), trans_normalize])
        transform_train = transforms.Compose([transforms.Resize((p[c.AUGMENT_RESIZE]), antialias=True), transforms.RandAugment(), transforms.ToTensor(), trans_normalize])
    else:
        # basic no augmentation
        transform_train = transforms.Compose([transforms.Resize((p[c.AUGMENT_RESIZE]), antialias=True), transforms.CenterCrop(p[c.AUGMENT_CROP]), transforms.ToTensor(), trans_normalize])

    #if p[c.LR_INCREASE]: transform_train = transforms.Compose([transforms.Resize((p[c.RESIZE], p[c.RESIZE]), antialias=True), transforms.ToTensor(), trans_normalize])
    transform_val = transforms.Compose([transforms.Resize((p[c.AUGMENT_RESIZE]), antialias=True), transforms.CenterCrop(p[c.AUGMENT_CROP]), transforms.ToTensor(), trans_normalize])
    transform_test = transforms.Compose([transforms.Resize((p[c.AUGMENT_RESIZE]), antialias=True), transforms.CenterCrop(p[c.AUGMENT_CROP]), transforms.ToTensor(), trans_normalize])

    print(str(transform_train))
    return transform_train, transform_val, transform_test

'''
transform_train = transforms.Compose([ #transforms.ToPILImage(),
                                  transforms.RandomRotation(20),
                                  #transforms.RandomResizedCrop((22,22)),
                                  transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                                  #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                  #transforms.RandomPerspective(),
                                  #transforms.Resize((28,28)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (1,))])

transform_training = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1,))])
transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1,))])


transform_train = transforms.Compose(
    [transforms.Resize((227,227)),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    trans_normalize])

transform_test = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), trans_normalize])
'''



######################              CRITERION              #######################

criterion = nn.CrossEntropyLoss()




######################              DATALOADERS              #######################

def dataloaders(p, train_ds, validate_ds, test_ds = None) :  # p is parameters
    # prepare dataloader
    train_dl = DataLoader(train_ds, batch_size=p[c.BATCH_SIZE], shuffle=True, num_workers=p[c.NUM_WORKERS], pin_memory=p[c.PIN_MEMORY])
    val_dl = DataLoader(validate_ds, batch_size=p[c.BATCH_SIZE], shuffle=False, num_workers=p[c.NUM_WORKERS], pin_memory=p[c.PIN_MEMORY])
    #test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = None
    if test_ds != None:
        test_dl = DataLoader(test_ds, batch_size=p[c.BATCH_SIZE], shuffle=False, num_workers=p[c.NUM_WORKERS], pin_memory=p[c.PIN_MEMORY])

    return train_dl, val_dl, test_dl




######################              NORMALIZATION              #######################

def transform_normalize_imagenet():
    transform_imagenet = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform_imagenet

def transform_normalize_cifar10():
    transform_imagenet = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    return transform_imagenet
