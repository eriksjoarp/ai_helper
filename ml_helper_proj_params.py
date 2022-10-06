import random
import time
from . import constants_ai_h as c

def params():
    project_params_dict = {}

    # experiment
    project_params_dict[c.DEVICE] = c.DEVICE_1
    project_params_dict[c.START_TIME] = int(time.time())
    project_params_dict[c.EXPERIMENT_SEED] = str(random.randint(0, 1000000))
    project_params_dict[c.MODEL_NAME_BEST_OLD] = False
    project_params_dict[c.LOGFILE] = []

    # model
    project_params_dict[c.MODEL_NAME] = 'mobilenet_v3_small_224'

    # training
    project_params_dict[c.BATCH_SIZE] = 500
    project_params_dict[c.EPOCHS] = 100
    project_params_dict[c.LR] = 0.0001
    project_params_dict[c.LR_FACTOR] = 100
    project_params_dict[c.LR_INCREASE] = True

    # transforms
    project_params_dict[c.AUGMENT_CROP] = 112
    project_params_dict[c.AUGMENT_RESIZE] = 112 # int(project_params_dict[c.AUGMENT_CROP] * 1.15)
    project_params_dict[c.AUGMENT] = c.AUGMENT_RANDAUGMENT
    #project_params_dict[c.AUGMENT] = c.AUGMENT_NONE

    # optimizer
    project_params_dict[c.OPTIMIZER] = c.OPTIMIZER_ADAM
    project_params_dict[c.OPTIMIZER_WEIGHT_DECAY] = 0.1

    # loss
    project_params_dict[c.CRITERION] = c.CRITERION_CROSS_ENTROPY

    # torch
    project_params_dict[c.RANDOM_SEED] = c.RANDOM_SEED_VALUE
    project_params_dict[c.CUDNN_BENCHMARK] = True
    project_params_dict[c.NUM_WORKERS] = 2
    project_params_dict[c.PIN_MEMORY] = True
    #project_params_dict[c.DEVICE] = c.EMPTY

    # results
    project_params_dict[c.ACC_BEST] = 0.0
    #project_params_dict[c.ACC2_BEST] = 0.0
    #project_params_dict[c.ACC3_BEST] = 0.0
    #project_params_dict[c.ACC4_BEST] = 0.0
    #project_params_dict[c.F1_BEST] = 0.0
    project_params_dict[c.ACC_BEST_EPOCHE] = 0

    # data
    project_params_dict[c.TRAIN_DS] = c.NONE
    project_params_dict[c.VAL_DS] = c.NONE
    project_params_dict[c.TEST_DS] = c.NONE
    project_params_dict[c.TRAIN_DL] = c.NONE
    project_params_dict[c.VAL_DL] = c.NONE
    project_params_dict[c.TEST_DL] = c.NONE

    project_params_dict[c.LOGFILE_NAME] = project_params_dict[c.MODEL_NAME] + '_IMG_SIZE_' + str(project_params_dict[c.AUGMENT_CROP]) + '_' + str(random.randint(0, 100000)) + '.log'

    #project_params_dict[] =
    return project_params_dict
