import torch
import time
import torch.optim as optim
import ml_helper_training
from ..helper import erik_functions_files

import constants_ai_h as c
import os
import torch_help_functions
import torch_help
import torchvision



def image_trainer(model, train_dl, val_dl, test_dl, criterion, optimizer, lr=0.0001, lr_scheduler = 'learning_rate_change', epochs = 20, debug = False):
    # preparing
    if debug: torch_help_functions.is_cuda_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)

    start_time = time.time()
    lr_new = lr

    # training loop
    for epoch in range(epochs):
        loss_ep = 0

        if lr_scheduler == 'learning_rate_change':
            lr_new = ml_helper_training.learning_rate_change(lr, epoch, 10, 0.33, 2)    # ToDo make new function
            optimizer = optim.Adam(model.parameters(), lr=lr_new)

        for batch_idx, (data, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward pass
            optimizer.zero_grad()
            scores = model(data)

            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()

        print(f'Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}')

        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data, targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)

                # forward pass
                scores = model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

            running_time = int(time.time() - start_time)

            print(f'Got {num_correct / num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.3f} duration {running_time} seconds lr:{lr_new}')


#ToDo make sure p is updated properly and passed back
def model_validate(p, val_dl, model, epoch, criterion, lr_new, device):
    # validate model
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        loss_ep = 0

        for batch_idx, (inputs, targets) in enumerate(val_dl):
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            #criterion = criterion.to(device=device)

            # forward pass
            with torch.cuda.amp.autocast():
                output = model(inputs).squeeze()
                loss = criterion(output, targets)

                _, predictions = output.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

            loss = loss.detach().cpu()  # .cpu().numpy()
            loss_ep = loss_ep * 0.9 + loss.item() * 0.1

        running_time = int(time.time() - p[c.START_TIME])

        print_update = f'Accuracy, {float(num_correct) / float(num_samples) * 100:.3f}, seconds, {running_time}, lr,{lr_new} ,loss,{loss_ep:.4f}'
        print(print_update)
        p[c.LOGFILE].append(print_update)

        # save best model
        # PARAMETER
        if not p[c.LR_INCREASE]:
            if float(num_correct) / float(num_samples) * 100.0 > p[c.ACC_BEST]:
                p[c.ACC_BEST] = float(num_correct) / float(num_samples) * 100.0

                model_name = p[c.MODEL_NAME] + '_acc_' + str(p[c.ACC_BEST]) + '_duration_' + str(
                    running_time) + '_experiment_nr' + str(p[c.EXPERIMENT_SEED]) + '.pth'
                path = os.path.join(c.DIR_MODELS_SAVE, model_name)
                torch.save(model.state_dict(), path)

                p[c.LOGFILE].append('NEW_BEST_ACCURACY,' + str(p[c.ACC_BEST]) + ',' + c.EPOCH + ',' + str(epoch))

                if p[c.MODEL_NAME_BEST_OLD]:
                    try:
                        os.remove(os.path.join(os.path.join(c.DIR_MODELS_SAVE, p[c.MODEL_NAME_BEST_OLD])))
                    except:
                        pass
                p[c.MODEL_NAME_BEST_OLD] = model_name


def model_train(p):
    # training
    #p[c.DEVICE] = ('cuda' if torch.cuda.is_available() else 'cpu')
    devices = torch_help_functions.available_gpus()

    print('GPUs found : ' + str(len(devices)))
    device = torch.device(p[c.DEVICE])
    device = torch.device(p[c.DEVICE] if torch.cuda.is_available() else 'cpu')

    p[c.LOGFILE].append(p[c.DEVICE])
    p[c.LOGFILE].append(c.DEVICE + ',' + str(p[c.DEVICE]))

    # PARAMETER

    #model = torchvision.models.alexnet()  # , dropout=0.001)
    #t_h.model_replace_last_layer3(model, 666, c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, True)

    #model_type = 'alexnet'
    #model = t_h.model_create(model_type, 0, 66)

    # FINDME
    model = torchvision.models.mobilenet_v3_large(pretrained=True)

    # def transfer_learning(model, num_outputs, layer_type_last_new = c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, layers_with_no_grad = 1,debug = False):
    torch_help.transfer_learning(model, 10, c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, 3, True)

    #t_h.model_replace_last_layer3(model, 10, c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, True)

    #models, all_models = t_h_f.torchvision_image_models()

    #for i in all_models:
    #    print(str(i))

    #_, _ = t_h_f.model_parameters(model)

    #def model_replace_last_layer3(model, num_outputs, layer_type_last_new = c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, debug = False):
    #t_h.model_replace_last_layer3(model, 10, c.MODEL_OUTPUT_LAYER_TYPE_LINEAR, True)

    #print(model)

    torch_help_functions.model_parameters(model, True)
    print(type(model))

    model = model.to(device=device)

    # optimizer
    optimizer = torch_help.optimizer_load(p, model.parameters(), p[c.LR])

    # loss
    criterion = torch_help.criterion_load(p)

    p[c.LOGFILE].append(c.START_TIME + ',' + str(p[c.START_TIME]))

    scaler = torch.cuda.amp.GradScaler()

    # training loop
    for epoch in range(p[c.EPOCHS]):
        model.train()
        # PARAMETER
        #   learning_rate_change2(learning_rate_start, epoch_now,   epochs_change , factor_change = 0.33,   warmup_epoch_levels = 4, warmup_epochs_per_level = 3, warmpup_factor_change = 2, warmup_factor_lower_than_max_lr = 20, learning_rate_max_test = False):
        #lr_new = ai_h.learning_rate_change2(p[c.LR], epoch, 4, 0.75, 2 * 5, 1, 1.41, 40, p[c.LR_INCREASE])
        #lr_new = ai_h.learning_rate_change2(p[c.LR], epoch, 4, 0.75, 2 * 5, 1, 1.41, 40, p[c.LR_INCREASE])

        # learning_rate_exponential(epochs, epoch_now,      learning_rate_max, factor,      warm_up_epochs=10, epochs_plateau=5,        epochs_linear=40, learning_rate_max_test=False):
        lr_new = ml_helper_training.learning_rate_exponential(p[c.EPOCHS], epoch, p[c.LR], p[c.LR_FACTOR], 3, 3, 10, p[c.LR_INCREASE])
        #lr_new = p[c.LR]

        if epoch == 0: torch_help_functions.layers_no_grad(model, 10)
        #if epoch == 3: t_h_f.layers_no_grad(model, 5)
        #if epoch == 10: t_h_f.layers_no_grad(model, 10)

        torch_help_functions.layer_requires_grad(model)



        #torch_help.learning_rate_optimizer_update(optimizer, lr_new)  # what does it do?
        loss_ep = 0

        for batch_idx, (inputs, target) in enumerate(train_dl):
            inputs = inputs.to(device=device)
            target = target.to(device=device)
            #criterion = criterion.to(device=device)

            # forward pass
            optimizer.zero_grad()

            # use fp16 instead of fp32 automatically etc
            with torch.cuda.amp.autocast():
                output = model(inputs).squeeze()
                #output = model(inputs)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            loss = loss.detach().cpu()

            # scaler.step() first unscales the gradients of the optimizers assigned params
            scaler.step(optimizer)

            # updates the scaler for next iteration
            scaler.update()

            loss_ep = loss_ep * 0.9 + loss.item() * 0.1 # item()

        # validation run
        model_validate(p, model, epoch, criterion, lr_new, device)

        print_update = f'Loss in epoch {epoch} , {loss_ep:.4f}'
        print(print_update)
        p[c.LOGFILE].append(print_update)

        # write logfile to disk
        path = os.path.join(c.DIR_LOGS, p[c.LOGFILE_NAME])
        if not erik_functions_files.write_list_to_file(path, p[c.LOGFILE]):
            print('ERROR could not print logfile')
