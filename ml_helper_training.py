import os
import torch
from . import constants_ai_h as c
import matplotlib.pyplot as plt
import math


def learning_rate_change2(learning_rate_start, epoch_now, epochs_change , factor_change = 0.33, warmup_epoch_levels = 4, warmup_epochs_per_level = 3, warmpup_factor_change = 2, warmup_factor_lower_than_max_lr = 20, learning_rate_max_test = False):
    starting_epoche = epoch_now - warmup_epoch_levels * warmup_epochs_per_level
    factor_times = starting_epoche // epochs_change
    warmpup_level = epoch_now // warmup_epochs_per_level

    # lr if not in warmup phase
    learning_rate_new = learning_rate_start * (factor_change ** factor_times)

    # warmup
    if starting_epoche < 0:
        learning_rate_new = learning_rate_start / warmup_factor_lower_than_max_lr * (warmpup_factor_change ** warmpup_level)

    #print(str(epoch_now) + ' ' + str(learning_rate_new) + ' ' + str(warmpup_level))
    if learning_rate_max_test: learning_rate_new = learning_rate_find_max(learning_rate_start, epoch_now, 1.1)
    return learning_rate_new


def learning_rate_exponential(epochs, epoch_now, learning_rate_max, factor, warm_up_epochs = 10, epochs_plateau = 5, epochs_linear = 40, learning_rate_max_test = False):
    # warmpup
    factor_linear = 10

    if epoch_now < warm_up_epochs:
        exponential_factor = factor ** (1.0/warm_up_epochs)
        lr_save = (1 / (exponential_factor ** (warm_up_epochs - epoch_now)))
        lr_new = lr_save * learning_rate_max
    elif epoch_now < (warm_up_epochs + epochs_plateau):
        lr_new = learning_rate_max
    elif epoch_now < (warm_up_epochs + epochs_plateau + epochs_linear):
        epochs_in_phase = epoch_now - warm_up_epochs - epochs_plateau
        lr_new = learning_rate_max * (1.1 - epochs_in_phase / epochs_linear)
    else:
        epochs_in_phase = epochs - warm_up_epochs - epochs_plateau - epochs_linear
        epoch_in_phase = epoch_now - warm_up_epochs - epochs_plateau - epochs_linear
        exponential_factor = (factor / factor_linear) ** (1.0 / (epochs - warm_up_epochs - epochs_plateau - epochs_linear))
        lr_new = learning_rate_max / factor_linear / (exponential_factor ** epoch_in_phase)

    # if learning rate max test
    if learning_rate_max_test: lr_new = learning_rate_find_max(learning_rate_max, epoch_now, 1.1)
    return lr_new

def learning_rate_find_max(lr_max, epoch_now, increase = 1.2):
    lr_increase = lr_max * (1.1 ** (epoch_now - 1))
    if epoch_now < 1:
        lr_increase = lr_max / 20
    return lr_increase

def gpu_memory(num_gpus = 2):
    gpu_mem = []
    for i in range(2):
        command = f'nvidia-smi -i {i} --query-gpu=utilization.memory --format=csv'
        txt = str(os.system(command))
        mem = int(txt * 100)
        gpu_mem.append(str(mem))
    return gpu_mem

def learning_rate_step(learning_rate_old, epoch, step, factor):
    size = epoch // step
    learning_rate_new = learning_rate_old * (factor ** size)
    return learning_rate_new

def learning_rate_up_and_down(learning_rate_start, epoch_now, epochs_total, max_factor):
    step = max_factor / ((epochs_total - 10 ) / 2)
    if epoch_now < 5:
        learning_rate_new = learning_rate_start
    elif epoch_now < epochs_total / 2 : # warming up
        learning_rate_new = step * (epoch_now - 5) * learning_rate_start + learning_rate_start
    elif epoch_now < epochs_total - 5 : # warming up
        learning_rate_new = step * (epochs_total - (epoch_now + 5)) * learning_rate_start + learning_rate_start
    else:
        learning_rate_new = learning_rate_start

    return learning_rate_new


# define some helper functions
def get_item(preds, labels):
    """function that returns the accuracy of our architecture"""
    return preds.argmax(dim=1).eq(labels).sum().item()


# turn off gradients during inference for memory effieciency
def get_all_preds(network, dataloader):
    """function to return the number of correct predictions across data set"""
    all_preds = torch.tensor([])
    model = network
    for batch in dataloader:
        images, labels = batch
        preds = model(images)  # get preds
        all_preds = torch.cat((all_preds, preds), dim=0)  # join along existing axis

    return all_preds


# save best model
def save_best_model(lr_increase, num_correct, num_samples, acc_best, model_name, running_time, model, logfile, model_name_best_old, epoch):      #   ToDo    fix parameters
    if not lr_increase:
        if float(num_correct) / float(num_samples) * 100 > acc_best:
            acc_best = float(num_correct) / float(num_samples) * 100
            model_name = model_name + '_acc_' + str(acc_best) + '_duration_' + str(running_time) + '.pth'
            path = os.path.join(c.DIR_MODELS_SAVE, model_name)
            torch.save(model.state_dict(), path)
            logfile.append('NEW_BEST_ACCURACY,' + str(acc_best) + ',EPOCH,' + str(epoch))
            if model_name_best_old:
                try:
                    os.remove(os.path.join(os.path.join(c.DIR_MODELS_SAVE, model_name_best_old)))
                except:
                    pass
            model_name_best_old = model_name


#########       cosine annealing
def lr_cosine_annealing(steps_tot, epochs):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    lrs = []

    steps_epoch = int(steps_tot / epochs)
    for epoch in range(epochs):
        for i in range(steps_epoch):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            #     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
            scheduler.step()

    plt.plot(lrs)
    plt.show()


#########           1cycle
def lr_1cycle(epochs, steps_per_epoch):
    model = torch.nn.Linear(2, 1)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.3)
    lrs = []

    for epoch in range(epochs):
        for i in range(steps_per_epoch):
            optimizer.step()
            lr_now = math.log(optimizer.param_groups[0]["lr"])
            lrs.append(lr_now)
            #print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
            scheduler.step()

    plt.plot(lrs)
    plt.show()

    print(lrs[0])
    print(lrs[len(lrs)-1])
    print(10 ** lrs[0])
    print(10 ** lrs[len(lrs)-1])