import torch
import torchvision
import constants as c

def is_cuda_available():
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        print('CUDA is available')
        print('GPUs ' + str(torch.cuda.device_count()))
        print(torch.cuda.get_device_name(0))

        print('__CUDA VERSION:', torch.version.cuda)
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
    else:
        print('Cannot find GPU')

def transform_normalize_imagenet():
    transform_imagenet = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform_imagenet

def transform_normalize_cifar10():
    transform_imagenet = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    return transform_imagenet

def available_gpus():
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    return available_gpus

def model_parameters(model, debug = False):
    layernames_modules = ([n for n, _ in model.named_modules()])
    layernames_children = ([n for n, _ in model.named_children()])
    #print(torchsummary.summary(model, (3, 224, 224)))

    if debug:
        print(model)
        print('layernames_modules')
        names = 0
        for name in layernames_modules:
            names +=1
            print(str(names) + ' ' + name)

        print('\nlayernames_children\n')
        names = 0
        for name in layernames_children:
            names += 1
            print(str(names) + ' ' + name)

    return layernames_children, layernames_children

def torchvision_image_models():
    torchvision_modules = (dir(torchvision.models))
    modules = []
    modules_all = []
    for module in torchvision_modules:
        if module[0].isupper():
            modules.append(module)
        if module[0] != '_':
            modules_all.append(module)
    return modules, modules_all

def model_type(model, debug = False):
    model_type = str(type(model))
    for model_name in c.MODEL_TORCHVISION_FC_TYPES:
        if model_name in model_type:
            return c.MODEL_IS_FC_TYPE
    for model_name in c.MODEL_TORCHVISION_CLASSIFIER_TYPES:
        if model_name in model_type:
            return c.MODEL_IS_CLASSIFIER_TYPE
    for model_name in c.MODEL_TORCHVISION_CLASSIFIER_DEEP_TYPES:
        if model_name in model_type:
            return c.MODEL_IS_CLASSIFIER_DEEP_TYPE
    if debug: print('could not match ' + model_type + ' to a specific type')
    return c.MODEL_TYPE_UNKNOWN

def torchvision_image_models2():
    models_all = []
    for model_list in c.MODELS_TORCHVISION:
        for model_item in model_list:
            models_all.append(model_item.lower())
    return models_all

def layers_no_grad(model, layers_with_no_grad = 1):
    layers_total = 0
    layer_now = 0
    for param in model.parameters():
        param.requires_grad = False
        layers_total +=1
    for param in model.parameters():
        if (layers_total - layer_now) <= layers_with_no_grad:
            param.requires_grad = True
        layer_now += 1


def layer_requires_grad(model):
    i = 0
    for param in model.parameters():
        i += 1
        if param.requires_grad == True:
            print(str(i) + ' requires grad')
    print(str(i) + ' layers')


if __name__ == "__main__":
    models_all = torchvision_image_models2()
    models_all2 = torchvision_image_models()

    model_parameters(torchvision.models.mobilenet_v3_large())
    print(torchvision.models.mobilenet_v3_large())

    model = torchvision.models.mobilenet_v3_large()

    for param in model.parameters():
        pass
        #print(param())
