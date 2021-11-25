from models.resnet_simclr import ResNetSimCLR as SSLResNet

DATA_ARGS = {"cifar10": {"kwargs": {"num_classes": 10}},
             "imbalanced_cifar10": {"kwargs": {"num_classes": 10}},
             "imagenet": {"kwargs": {"num_classes": 1000}},
             "imbalanced_imagenet": {"kwargs": {"num_classes": 1000}}, }
MODEL_ARGS = {"SSLResNet18": {"net": SSLResNet, "args": ['resnet18']},
              "SSLResNet50": {"net": SSLResNet, "args": ['resnet50']},}


def get_net_by_dataset(data_name, model_name):
    """
    Args:
        data_name(str): The name of the dataset.

    Returns: Semantic Segmentation Network.

    """

    model_args = MODEL_ARGS[model_name]["args"]
    model_kwargs = DATA_ARGS[data_name]["kwargs"]
    net = MODEL_ARGS[model_name]["net"]

    net = net(*model_args, **model_kwargs)
    return net


def get_networks(data_name, model_name):
    return get_net_by_dataset(data_name, model_name)
