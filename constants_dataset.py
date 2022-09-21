import os, sys
import constants_ai_h as c_ai_h

sys.path.append(c_ai_h.DIR_HELPER)
import helpers as h

#######################             Directorys

BASE_DIR_GIT = c_ai_h.BASE_DIR_GIT
BASE_DIR_DATASET = h.j(c_ai_h.BASE_DIR_AI, 'datasets')

DIR_DATASET_CTU13 = h.j(BASE_DIR_DATASET, r'ctu13\CTU-13-Dataset')
DIR_DATASET_EUROSAT = h.j(BASE_DIR_DATASET, r'eurosat')



#######################             Download links

URL_CTU13 = r'''https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2'''  # Download and extract the data into your DATASET_BASEDIR



#######################

DATASETS_ALL = 'DATASETS_ALL'
ROWS_ALL = 'ROWS_ALL'

CONCATENATE_TRUE = 'True'
CONCATENATE_FALSE = 'False'