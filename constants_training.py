

#########################                 huggingface vision models                #################################
#
#   swinv2 config
#

S2_PATH_CONFIG = 's2_path_config'
HUGGINGFACE_AUTHENTICATE_TOKEN = 'huggingface_authenticate_token'

S2_MODEL_CHECKPOINT = 's2_model_checkpoint'
S2_MODEL_NAME = 's2_model_name'
S2_MODEL_NAME_TRAINED = 's2_model_name_trained'
S2_MODEL_NAME_TRAINED_PRE_CHOSEN = 's2_model_name_trained_pre_chosen'

S2_SAVE_DATA_DIR = 's2_save_data_dir'
S2_MODEL_TRAIN = 's2_model_train'
S2_MODEL_EVALUATE = 's2_model_evaluate'

# training_arguments
#S2_TRAIN_SAVE_DATA_DIR = 's2_train_save_data_dir'
S2_TRAIN_ARGS_LR = 's2_train_args_lr'
S2_TRAIN_ARGS_NUM_TRAIN_EPOCHS = 's2_train_args_num_train_epochs'
S2_TRAIN_ARGS_WEIGHT_DECAY = 's2_train_args_weight_decay'

S2_TRAIN_ARGS_WARMUP_RATIO = 's2_train_args_warmup_ratio'
S2_TRAIN_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE = 's2_train_args_per_device_train_batch_size'
S2_TRAIN_ARGS_PER_DEVICE_EVAL_BATCH_SIZE= 's2_train_args_per_device_eval_batch_size'
S2_TRAIN_ARGS_GRADIENT_ACCUMULATION_STEPS = 's2_train_args_gradient_accumulation_steps'
S2_TRAIN_ARGS_DATALOADER_NUM_WORKERS = 's2_train_args_dataloader_num_workers'

S2_TRAIN_ARGS_PUSH_TO_HUB = 's2_train_args_push_to_hub'
S2_TRAIN_ARGS_TEST_SIZE = 's2_train_args_test_size'
S2_TRAIN_ARGS_REMOVE_UNUSED_COLUMNS = 's2_train_args_remove_unused_columns'
S2_TRAIN_ARGS_EVALUATION_STRATEGY = 's2_train_args_evaluation_strategy'
S2_TRAIN_ARGS_SAVE_STRATEGY = 's2_train_args_save_strategy'

S2_TRAIN_ARGS_LOGGING_STEPS = 's2_train_args_logging_steps'
S2_TRAIN_ARGS_LOAD_BEST_MODEL_AT_END = 's2_train_args_load_best_model_at_end'
S2_TRAIN_ARGS_METRIC_FOR_BEST_MODEL = 's2_train_args_metric_for_best_model'
S2_TRAIN_ARGS_SEED = 's2_train_args_seed'
S2_TRAIN_ARGS_OPTIM = 's2_train_args_optim'
S2_TRAIN_ARGS_BF16 = 's2_train_args_bf16'

# trainer
S2_TRAINER_TRAIN_DATASET = 's2_trainer_train_dataset'
S2_TRAINER_EVAL_DATASET = 's2_trainer_eval_dataset'
S2_TRAINER_TEST_DATASET = 's2_trainer_test_dataset'
S2_TRAINER_CONFUSION_MATRIX_SAVE = 's2_trainer_confusion_matrix_save'
S2_TRAINER_COMPUTE_METRICS = 's2_trainer_compute_metrics'
S2_TRAINER_DATA_COLLATOR = 's2_trainer_data_collator'

# trainer metrics
S2_TRAINER_METRICS_ACCURACY = 's2_trainer_metrics_accuracy'
S2_TRAINER_METRICS_F1 = 's2_trainer_metrics_f1'
S2_TRAINER_METRICS_PRECISION = 's2_trainer_metrics_precision'
S2_TRAINER_METRICS_RECALL = 's2_trainer_metrics_recall'



################              dataset config                  #################

DATASET_PATH_CONFIG = 'dataset_path_config'
DATASET_PATH_DATASET = 'dataset_path_dataset'
DATASET_SMALL = 'dataset_small'
DATASET_DIR_TRAINING_DATA_BASE = 'dataset_dir_training_data_base'
DATASET_DIR_TRAINING_DATA = 'dataset_dir_training_data'
DATASET_MAX_LABELS_PER_CLASS = 'dataset_max_labels_per_class'

# transforms
DATASET_TRAIN_TRANSFORMS = 'dataset_train_transforms'
DATASET_TRAIN_TRANSFORMS_RESIZE = 'dataset_train_transforms_resize'
DATASET_TRAIN_TRANSFORMS_CENTERCROP = 'dataset_train_transforms_centercrop'
DATASET_TRAIN_TRANSFORMS_TO_TENSOR = 'dataset_train_transforms_to_tensor'
DATASET_TRAIN_TRANSFORMS_NORMALIZE = 'dataset_train_transforms_normalize'

DATASET_VAL_TRANSFORMS = 'dataset_val_transforms'
DATASET_VAL_TRANSFORMS_RESIZE = 'dataset_val_transforms_resize'
DATASET_VAL_TRANSFORMS_CENTERCROP = 'dataset_val_transforms_centercrop'
DATASET_VAL_TRANSFORMS_TO_TENSOR = 'dataset_val_transforms_to_tensor'
DATASET_VAL_TRANSFORMS_NORMALIZE = 'dataset_val_transforms_normalize'

# default values
IS_TRUE = True
IS_FALSE = False
IS_DEFAULT = 'default'



