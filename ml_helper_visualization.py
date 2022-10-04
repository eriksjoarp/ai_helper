import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import matplotlib.patches as patches


# images as numpyarrays
def show_image_grid(images, grid_size_x, grid_size_y = False, labels = False, gray = False, permutate = True, plot_size=[15.00, 10.00]):
    if grid_size_y == False: grid_size_y = grid_size_x
    if len(images) < grid_size_y * grid_size_x:
        print('ERROR, too few images to show.')

    plt.rcParams["figure.figsize"] = plot_size
    plt.rcParams["figure.autolayout"] = True

    f, axarr = plt.subplots(grid_size_x, grid_size_y)

    counter = 0

    for x in range(grid_size_x):
        for y in range(grid_size_y):
            axarr[x, y].axis('off')
            if permutate:
                image_permutated = images[counter].permute(1, 2, 0)
            else:
                image_permutated = images[counter]

            if not(gray):
                axarr[x, y].imshow(image_permutated)
            else:
                axarr[x, y].imshow(image_permutated, cmap='gray')

            if labels:
                axarr[x, y].set_title(labels[counter], fontsize=12)
            counter += 1

    plt.show()



def un_normalize_image(image, show_image = False):
    new_image = image * 128
    new_image = new_image + 128

    new_image = np.transpose(new_image.cpu().numpy(), (1,2,0))
    if show_image:
        plt.imshow(image.numpy()[0], cmap='gray')
        plt.show()
    return new_image


def un_normalize_images(images):
    un_normalized_images = []
    for image in images:
        un_normalized_image = un_normalize_image(image)
        un_normalized_images.append(un_normalized_image)
    return un_normalized_images


def display_bounding_boxes(image, boxes, labels, confidences = 'None', path_save_to_disk = False, show = True):
    fig, ax = plt.subplots()
    ax.imshow(image)

    colors = ['r', 'pink', 'yellow', 'g','orange','black', 'brown', 'gray', 'blue', 'white']
    color_for_label = {}
    unique_labels = list(set(labels))
    for nr_label in range(len(unique_labels)):
        color_for_label[unique_labels[nr_label]] = colors[nr_label]

    for score, label, box in zip(confidences, labels, boxes):
        rect = patches.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=color_for_label[label], facecolor='none', label=label)
        ax.add_patch(rect)
        print(label + ' ' + color_for_label[label])

    if show:
        plt.show()
    if path_save_to_disk:
        plt.savefig(path_save_to_disk)
    return plt



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# images as tensors
def show_grid(images, labels):
    # create a grid
    plt.figure(figsize=(15,10))
    grid = torchvision.utils.make_grid(nrow=20, tensor=images)
    print(f"image tensor: {images.shape}")
    print(f"class labels: {labels}")
    plt.imshow(np.transpose(grid, axes=(1,2,0)), cmap='gray')
    plt.show()


def confusion_matrix_(net, testloader, path_save):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(path_save)


def plot_list(lista, show = True, save_path = None, label = ''):
    plt.plot(lista, color='magenta', marker='o',mfc='pink' ) #plot the data
    plt.xticks(range(0,len(lista)+1, 1)) #set the tick frequency on x-axis

    plt.ylabel('data') #set the label for y axis
    plt.xlabel('index') #set the label for x-axis
    plt.title(label) #set the title of the graph

    if save_path != None:
        plt.savefig(save_path)

    plt.show() #display the graph

