import os, sys
import constants_ai_h as c_ai_h

from os.path import join as j
from ..helper import helpers as h


#######################             Directorys

BASE_DIR_GIT = c_ai_h.BASE_DIR_GIT
BASE_DIR_DATASET = j(c_ai_h.BASE_DIR_AI, 'datasets')

DIR_AI_HELPER = h.j(BASE_DIR_GIT, 'ai_helper')

DIR_DATASET_CTU13 = j(BASE_DIR_DATASET, r'ctu13\CTU-13-Dataset')
DIR_DATASET_BASE_EUROSAT =  j(BASE_DIR_DATASET, 'eurosat')
DIR_DATASET_EUROSAT_RGB = j(DIR_DATASET_BASE_EUROSAT, 'EuroSAT', '2750')
DIR_DATASET_EUROSAT_MS = j(DIR_DATASET_BASE_EUROSAT, 'EuroSATallBAnds', '2750')




#######################             Download links

URL_CTU13 = r'''https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2'''  # Download and extract the data into your DATASET_BASEDIR



#######################

DATASETS_ALL = 'DATASETS_ALL'
ROWS_ALL = 'ROWS_ALL'

CONCATENATE_TRUE = 'True'
CONCATENATE_FALSE = 'False'
