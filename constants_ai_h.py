import pathlib
import os


BASE_DIR_AI = r'C:\ai'
BASE_DIR_PROJ = pathlib.Path(os.getcwd())
BASE_DIR_GIT = BASE_DIR_PROJ.parent.absolute()

# dataset directory
DIR_BASE_AI = r'C:\ai'
DIR_BASE_EXPERIMENTS = r'C:\ai_experiments'

DIR_EXPERIMENTS_SWIN = os.path.join(DIR_BASE_EXPERIMENTS, 'swin')

DIR_AI_HELPER = os.path.join(BASE_DIR_GIT, 'ai_helper')
DIR_HELPER = os.path.join(BASE_DIR_GIT, 'helper')

DIR_DATASETS = os.path.join(DIR_BASE_AI, 'datasets')
DIR_DATASETS_LARGE = r'D:\ai\datasets'
DIR_MODELS_SAVE = os.path.join(DIR_BASE_EXPERIMENTS, 'models_saved')
DIR_LOGS = os.path.join(DIR_BASE_EXPERIMENTS, 'logs')
DIR_RESULTS = os.path.join(DIR_BASE_EXPERIMENTS, 'results')

DIR_CONFIGS = os.path.join(DIR_BASE_EXPERIMENTS, 'configs')
DIR_AUGMENT_CONFIGS = os.path.join(DIR_CONFIGS, 'augment_configs')
DIR_CONFIGS_EXPERIMENTS_TO_RUN = os.path.join(DIR_CONFIGS, 'experiments_to_run')
DIR_CONFIGS_DEFAULTS = os.path.join(DIR_CONFIGS, 'configs_default')

DIR_CONFIGS_RUN = os.path.join(DIR_CONFIGS_EXPERIMENTS_TO_RUN, 'experiments_run')

'''
experiments folder: running number plus
config
result
log
model_saved
'''

RANDOM_STATE = 42

# datasets
MNIST_NUMBERS = 0

# train
doTrain = True
noTrain = False
NONE = None
EMPTY = 'EMPTY'

# experiment
LOGFILE_NAME = 'LOGFILE_NAME'
EXPERIMENT_SEED = 'EXPERIMENT_SEED'
LR_INCREASE = 'LR_INCREASE'
LR_FACTOR = 'LR_FACTOR'

# model
MODEL_TYPE = 'MODEL_TYPE'
MODEL_NAME = 'MODEL_NAME'
BATCH_SIZE = 'BATCH_SIZE'
EPOCHS = 'EPOCHS'
LR = 'LR'
START_TIME = 'START_TIME'
EPOCH = 'EPOCH'

MODEL_TYPE_CNN = 'MODEL_TYPE_CNN'
MODEL_TYPE_CLUSTERING = 'MODEL_TYPE_CLUSTERING'
MODEL_TYPE_ML = 'MODEL_TYPE_ML'

MODEL_OUTPUT_LAYER_TYPE = 'MODEL_OUTPUT_LAYER_TYPE'
MODEL_OUTPUT_LAYER_TYPE_LINEAR = 'MODEL_OUTPUT_LAYER_TYPE_LINEAR'
MODEL_OUTPUT_LAYER_TYPE_SIGMOID = 'MODEL_OUTPUT_LAYER_TYPE_SIGMOID'

MODEL_MOBILENET_V3 = 'MODEL_MOBILENET_V3'
MODEL_RESNET = 'MODEL_RESNET'

# transforms
AUGMENT = 'AUGMENT'
AUGMENT_CONFIG = 'AUGMENT_CONFIG'
AUGMENT_RESIZE = 'AUGMENT_RESIZE'
AUGMENT_CROP = 'AUGMENT_CROP'
AUGMENT_RANDAUGMENT = 'AUGMENT_RANDAUGMENT'
AUGMENT_NONE = 'AUGMENT_NONE'

# pytorch
NUM_WORKERS = 'NUM_WORKERS'
PIN_MEMORY = 'PIN_MEMORY'
CUDNN_BENCHMARK = 'CUDNN_BENCHMARK'
RANDOM_SEED_VALUE = 17
RANDOM_SEED = 'RANDOM_SEED'
DEVICE = 'DEVICE'
DEVICE_0 = 'cuda:0'
DEVICE_1 = 'cuda:1'
DEVICE_2 = 'cuda:2'

MODEL_NAME_BEST_OLD = 'MODEL_NAME_BEST_OLD'
LOGFILE = 'LOGFILE'

# optimizer
OPTIMIZER = 'OPTIMIZER'
OPTIMIZER_ADAM = 'OPTIMIZER_ADAM'
OPTIMIZER_ADAMW = 'OPTIMIZER_ADAMW'
OPTIMIZER_BETA1 = 'OPTIMIZER_BETA1'
OPTIMIZER_BETA2 = 'OPTIMIZER_BETA2'
OPTIMIZER_WEIGHT_DECAY = 'OPTIMIZER_WEIGHT_DECAY'

# loss functions
CRITERION = 'CRITERION'
CRITERION_CROSS_ENTROPY = 'CRITERION_CROSS_ENTROPY'
LOSS_DECAY = 'LOSS_DECAY'

# data
DATASET_NAME = 'DATASET_NAME'
DATASET_TYPE = 'DATASET_TYPE'
DATASET_SOURCE = 'DATASET_SOURCE'

TRAIN_DS = 'TRAIN_DS'
VAL_DS = 'VAL_DS'
TEST_DS = 'TEST_DS'
TRAIN_DL = 'TRAIN_DL'
VAL_DL = 'VAL_DL'
TEST_DL = 'TEST_DL'
TRAIN_TRANSFORM = 'TRAIN_TRANSFORM'
VAL_TRANSFORM = 'VAL_TRANSFORM'
TEST_TRANSFORM = 'TEST_TRANSFORM'

# results
ACC_BEST = 'ACC_BEST'
ACC_BEST_EPOCHE = 'ACC_BEST_EPOCHE'

# torchvision models
MODEL_IS_FC_TYPE = 'MODEL_IS_FC_TYPE'
MODEL_IS_CLASSIFIER_TYPE = 'MODEL_IS_CLASSIFIER_TYPE'
MODEL_IS_CLASSIFIER_DEEP_TYPE = 'MODEL_IS_CLASSIFIER_DEEP_TYPE'
MODEL_TYPE_UNKNOWN = 'MODEL_TYPE_UNKNOWN'

MODEL_TORCHVISION_FC_TYPES = [
    'GoogLeNet',
    'Inception3',
    'RegNet',
    'ShuffleNetV2',
    'ResNet',
    'ResNext'
]
MODEL_TORCHVISION_CLASSIFIER_TYPES = [
    'DenseNet',
    'NoNet'
]
MODEL_TORCHVISION_CLASSIFIER_DEEP_TYPES = [
    'AlexNet',
    'ConvNeXt',
    'EfficientNet',
    'MNASNet',
    'MobileNetV2',
    'MobileNetV3',
    'SqueezeNet',
    'VGG'
]

MODELS_TORCHVISION = [
    MODEL_TORCHVISION_FC_TYPES,
    MODEL_TORCHVISION_CLASSIFIER_TYPES,
    MODEL_TORCHVISION_CLASSIFIER_DEEP_TYPES
]


# others



