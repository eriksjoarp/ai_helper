import time
import torch
import sys
import os

class Experiment():
    def __init__(self, config_name):
        self.start_time = time.time()
        self.default_config = ''  # read default config and inherit from it
        self.config = ''  # read config and replace the default values with the new ones

        self.devices = self.available_gpus()

    def run_experiment(self):
        # do all things in order
        pass

    def log_write(self):
        pass

    def dataset_load(self):
        pass
        # create transformers

        # download dataset
        self.train_ds, self.val_ds, self.test_ds = 1, 1, 1

    def model(self):
        pass

    def train(self):
        pass

    def visualize(self):
        pass

    def create_experiment_base(self):
        pass
        # create dirs and log files and names etc
        # google keep track of ai projects

    def available_gpus(self):
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        self.devices = available_gpus



r'''
dataset:
  script_path: ../datasets/cifar10_keras.py
model:
  script_path: ../models/optimized.py
optimizer:
  script_path: ../optimizers/adam_keras.py
  initial_lr: 0.0001
train:
  script_path: ../train/train_keras.py
  artifacts_path: ../artifacts/cifar10_opt/
  batch_size: 64
  epochs: 1000
  data_augmentation:
    samplewise_center: False
    samplewise_std_normalization: False
    rotation_range: 0
    width_shift_range: 0.1
    height_shift_range: 0.1
    horizontal_flip: True
    vertical_flip: False
    zoom_range: 0
    shear_range: 0
    channel_shift_range: 0
    featurewise_center: False
    zca_whitening: False
evaluate:
  batch_size: 1000
  augmentation_factor: 32
  data_augmentation:
    samplewise_center: False
    samplewise_std_normalization: False
    rotation_range: 0
    width_shift_range: 0.15
    height_shift_range: 0.15
    horizontal_flip: True
    vertical_flip: False
    zoom_range: 0
    shear_range: 0
    channel_shift_range: 0
    featurewise_center: False
    zca_whitening: False

knn:
#INITIAL SETTINGS
data_directory: ../data/
data_name: breast-cancer-wisconsin.data
drop_columns: ["id"]
target_name: class
test_size: 0.2
model_directory: ../models/
model_name: KNN_classifier.pkl


#kNN parameters
n_neighbors: 5
weights: uniform
algorithm: auto
leaf_size: 15
p: 2
metric: minkowski
n_jobs: 1

'''