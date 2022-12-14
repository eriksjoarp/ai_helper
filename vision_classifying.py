'''
conda activate swin2
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge opencv -y
conda install -c huggingface transformers -y
conda install -c conda-forge huggingface_hub -y
conda install -c huggingface -c conda-forge datasets -y
conda install -c conda-forge pytorch-model-summary -y
conda install -c intel scikit-learn -y
conda install -c anaconda seaborn -y

'''

import os

from transformers import AutoFeatureExtractor, SwinForImageClassification

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torchvision.transforms as transforms

from helper import erik_functions_files
from ai_helper import ml_helper_visualization
from ai_helper import dataset_load_helper
from ai_helper import constants_dataset



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


def classify_swin(p, dataset, model_name, cache_dir = constants_dataset.DIR_MODEL_CACHE, dataset_path=False, show_grid=False, correct_labels=False , type_dataset = 'numpy', path_label_classes=constants_dataset.FILE_LABELS_IMAGENET1K):
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    DIR_MODEL_CACHE = constants_dataset.DIR_MODEL_CACHE

    if dataset_path:
        DIR_DATASET = os.path.join(constants_dataset.BASE_DIR_DATASET, 'ImageNet', 'test')

    #dataset = load_dataset("huggingface/cats-image", cache_dir=DIR_DATASET_HUGGINGFACE)
    #test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=p[c.NUM_WORKERS], pin_memory=p[c.PIN_MEMORY])

    # type of dataset ToDo: handle different types

    # load images and save the path to the images
    if dataset_path:
        images_path = dataset_path
        images, path_images_loaded = erik_functions_files.load_images_from_folder(images_path)
        images_nr = len(images)

    # prepare swin model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = SwinForImageClassification.from_pretrained(model_name, cache_dir=DIR_MODEL_CACHE)

    # test_dl = DataLoader(dataset['test'], batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    # predictions = get_all_preds(model, test_dl)

    label2id, id2label = dataset_load_helper.label_to_id(path_label_classes)

    pred_labels = []
    pred_labels_name = []

    with torch.no_grad():
        for i in range(images_nr):
            inputs = feature_extractor(images[i], return_tensors="pt")
            logits = model(**inputs).logits
            predicted_label = logits.argmax(-1).item()
            pred_labels.append(predicted_label)
            pred_labels_name.append(id2label[predicted_label])

    inputs = []
    for image in images:
        #input = torch.from_numpy(images)
        inputs.append(input)

    #   Compare to correct labels
    if correct_labels:                  #ToDo compare to correct labels
        pass

    if show_grid:
        ml_helper_visualization.show_image_grid(images, 4, permutate=False, labels=pred_labels_name)

    return pred_labels, pred_labels_name, path_images_loaded



