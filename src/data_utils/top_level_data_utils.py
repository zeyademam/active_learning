from .custom_cifar10 import get_data_cifar10
from .custom_imagenet import get_data_imagenet
from .custom_imbalanced_cifar10 import get_data_imbalanced_cifar10
from .custom_imbalanced_imagenet import get_data_imbalanced_imagenet


def get_data(data_path, data_name, supervised=False, debug_mode=False, imbalance_args=None, ):
    if data_name == "cifar10":
        return get_data_cifar10(data_path, input_size=(32, 32), supervised=supervised,
                                debug_mode=debug_mode, )
    elif data_name == 'imagenet':
        return get_data_imagenet(data_path, debug_mode=debug_mode, )
    elif data_name == 'imbalanced_cifar10':
        return get_data_imbalanced_cifar10(data_path, debug_mode=debug_mode,
                                           imbalance_args=imbalance_args, )
    elif data_name == 'imbalanced_imagenet':
        return get_data_imbalanced_imagenet(data_path)
    else:
        raise ValueError("Dataset does not exist")
