import torch
import torchvision

import ml_helper
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import constants_ai_h as c
import torch_help_functions


def datasets(action = 'list', module = 'torchvision'):
    if action == 'list':
        modules = torchvision.datasets
        for dataset in modules:
            print(dataset)

# torchvision datasets
def torchvision_datasets(print_names = False):
    dataset_names = dir(torchvision.datasets)
    dataset_names_return = []
    for dataset_name in dataset_names:
        if dataset_name[0].isupper():
            dataset_names_return.append(dataset_name)
            if print_names: print(dataset_name)
    return dataset_names_return

# number of parameters for a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# finds the mean and std of the dataset
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

# log parameters
def start_log_project_parameters(p):
    (p[c.LOGFILE]).append(c.RANDOM_SEED_VALUE + ',' + str(p[c.RANDOM_SEED_VALUE]))
    (p[c.LOGFILE]).append(c.BATCH_SIZE + ',' + str(p[c.BATCH_SIZE]))
    (p[c.LOGFILE]).append(c.EPOCHS + ',' + str(p[c.EPOCHS]))
    (p[c.LOGFILE]).append(c.LR + ',' + str(p[c.LR]))
    (p[c.LOGFILE]).append(c.AUGMENT_CROP + ',' + str(p[c.AUGMENT_CROP]))
    (p[c.LOGFILE]).append(c.LR_INCREASE + ',' + str(p[c.LR_INCREASE]))
    (p[c.LOGFILE]).append(c.MODEL_NAME + ',' + str(p[c.MODEL_NAME]))
    (p[c.LOGFILE]).append(c.NUM_WORKERS + ',' + str(p[c.NUM_WORKERS]))
    (p[c.LOGFILE]).append(c.PIN_MEMORY + ',' + str(p[c.PIN_MEMORY]))
    (p[c.LOGFILE]).append(c.LOGFILE_NAME + ',' + str(p[c.LOGFILE_NAME]))

# load an optimizer
def optimizer_load(p, model_params, lr_update):
    if p[c.OPTIMIZER] == c.OPTIMIZER_ADAM:
        return optim.Adam(model_params, lr=lr_update, weight_decay=p[c.OPTIMIZER_WEIGHT_DECAY])
    if p[c.OPTIMIZER] == c.OPTIMIZER_ADAMW:
        return optim.AdamW(model_params, lr=lr_update, weight_decay=p[c.OPTIMIZER_WEIGHT_DECAY],betas=(0.95, 0.99))
        #return optim.Adam(model_params, lr=lr_update)

# update the optimizer
def optimizer_update_lr(p, optimizer, lr_update):
    for g in optimizer.param_groups:
        g['lr'] = lr_update

# default criterion for vision models
def criterion_load(p):
    if p[c.CRITERION] == c.CRITERION_CROSS_ENTROPY:
        return nn.CrossEntropyLoss()

#   def model_replace_last_layer3(model, model_name, num_outputs, layer_type_last_new = c.MODEL_OUTPUT_LAYER_TYPE_LINEAR):
def transfer_learning(model, num_outputs, layer_type_last_new = c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, layers_with_no_grad = 1,debug = False):
    # set all layers to non-learning

    torch_help_functions.layers_no_grad(model, layers_with_no_grad)

    model = model_replace_last_layer3(model, num_outputs, layer_type_last_new)

    # add a new output layer with the number of classes you need
    #layernames = ([n for n, _ in model.named_modules()])
    #layernames = ([n for n, _ in model.named_children()])
    #last_layername = layernames[-1]

    #last_layer = getattr(model, last_layername)
    #num_features = last_layer[-1].in_features

    print(type(model))
    return model

# find the last layer in a model, used for replacing it with a new one with number of classifiers
def model_last_layer(model):
    # find last layer
    #layernames = ([n for n, _ in model.named_modules()])
    layernames = ([n for n, _ in model.children()])
    lastlayer_name = layernames[-1]

    lastlayer = getattr(model, lastlayer_name)
    return lastlayer

# create a new last layer
def layer_last_create(layer_type_last_new, num_outputs, num_inputs, bias):
    if layer_type_last_new == c.MODEL_OUTPUT_LAYER_TYPE_LINEAR:
        layer = nn.Linear(num_inputs, num_outputs, bias)
    elif layer_type_last_new == c.MODEL_OUTPUT_LAYER_TYPE_SIGMOID:
        layer = nn.Sigmoid()
    else:
        return False
    return  layer

# replaces the last layer
def model_replace_last_layer3(model, num_outputs, layer_type_last_new = c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, debug = False):
    model_type = torch_help_functions.model_type(model, debug)

    if model_type == c.MODEL_IS_CLASSIFIER_DEEP_TYPE:
        last_item_index = len(model.classifier) - 1
        old_fc = model.classifier.__getitem__(last_item_index)
        new_fc = nn.Linear(in_features=old_fc.in_features, out_features=num_outputs, bias=True)
        model.classifier.__setitem__(last_item_index, new_fc)
    elif model_type == c.MODEL_IS_FC_TYPE:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=num_outputs, bias=True)
    elif model_type == c.MODEL_IS_CLASSIFIER_TYPE:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features=in_features, out_features=num_outputs, bias=True)
    elif model_type == c.MODEL_TYPE_UNKNOWN:
        return False

    if debug:
        print('model ' + str(model))
        print('type of the model ' + str(type(model)))
    return True

# sets all layers to non trainable
def transfer_learning_lr(model, lr):
    current_layer = 0
    for param in model.parameters():
        if lr[current_layer] == False:
            param.requires_grad = False


# create a new model with the correct number of outputs
def model_create(model_type, model_nr, num_outputs = 10, pretrained = False, debug = False) :
    model_get = getattr(torchvision.models, model_type)
    model_complete = model_get(pretrained=pretrained)
    model_replace_last_layer3(model_complete, num_outputs, c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, debug)
    return model_complete






