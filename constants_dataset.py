import os
from ai_helper import constants_ai_h as c_ai_h

from os.path import join as j

#######################             Directorys

BASE_DIR_GIT = c_ai_h.BASE_DIR_GIT
BASE_DIR_MODEL_WEIGHTS = j(c_ai_h.BASE_DIR_AI, 'model_weights')
BASE_DIR_DATASET = j(c_ai_h.BASE_DIR_AI, 'datasets')
BASE_DIR_SR3 = os.path.join(BASE_DIR_DATASET, 'sr3')

DIR_DATASET_CTU13 = j(BASE_DIR_DATASET, r'ctu13\CTU-13-Dataset')
DIR_DATASET_BASE_EUROSAT =  j(BASE_DIR_DATASET, 'eurosat')
DIR_MODELS_SAVED = j(c_ai_h.BASE_DIR_AI, 'models_saved')

DIR_DATASET_EUROSAT_RGB = j(DIR_DATASET_BASE_EUROSAT, 'EuroSAT', '2750')
DIR_DATASET_EUROSAT_RGB_32 = j(DIR_DATASET_BASE_EUROSAT, 'EuroSAT', '2750_32')
DIR_DATASET_EUROSAT_RGB_16 = j(DIR_DATASET_BASE_EUROSAT, 'EuroSAT', '2750_16')
DIR_DATASET_EUROSAT_MS = j(DIR_DATASET_BASE_EUROSAT, 'EuroSATallBAnds', '2750')

DIR_DATASET_HUGGINGFACE = j(BASE_DIR_DATASET, 'huggingface')
DIR_LABELS = j(BASE_DIR_DATASET, 'labels_mapping')
DIR_DATASET_IMDB = j(DIR_DATASET_HUGGINGFACE, 'imdb')

DIR_MODEL_CACHE = j(BASE_DIR_GIT, 'model_cache')
DIR_DATASET_CACHE = j(BASE_DIR_DATASET, 'dataset_cache')

FILE_LABELS_IMAGENET1K = j(DIR_LABELS, 'imagenet1k.txt')
FILE_LABELS_IMAGENET22K = j(DIR_LABELS, 'imagenet22k.txt')
FILE_LABELS_EUROSAT = j(DIR_LABELS, 'eurosat.txt')

#######################             model weights

DIR_MODEL_WEIGHTS_GAN = j(BASE_DIR_MODEL_WEIGHTS, 'gan')
DIR_MODEL_WEIGHTS_GFPGAN = j(DIR_MODEL_WEIGHTS_GAN, 'gfp_gan')



#######################             Download links

URL_CTU13 = r'''https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2'''  # Download and extract the data into your DATASET_BASEDIR

URLS_WEIGHTS_GFP_GAN = [
    r'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    r'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth',
    r'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    r'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    r'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
    r'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth'
]


#######################             models saved

DIR_MODELS_SAVED_SWIN = j(DIR_MODELS_SAVED, 'swin')




#######################

DATASETS_ALL = 'DATASETS_ALL'
ROWS_ALL = 'ROWS_ALL'

CONCATENATE_TRUE = 'True'
CONCATENATE_FALSE = 'False'

DATASET_SMALL = '_small'
DATASET_SPLIT = '_split'

DATASET_TRAIN = 'train'
DATASET_VAL = 'val'
DATASET_TEST = 'test'



#########################################           SR3             ##########################################
DIR_DATASET_SR3_ERIK = j(BASE_DIR_SR3, 'erik')
DIR_DATASET_SR3_ERIK_PREPARED = j(BASE_DIR_SR3, 'erik_prepared')
DIR_DATASET_SR3_EUROSAT = j(BASE_DIR_SR3, 'eurosat')
DIR_DATASET_SR3_EUROSAT_PREPARED = j(BASE_DIR_SR3, 'eurosat_prepared')

