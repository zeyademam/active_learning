import logging
import os

from torchvision import datasets, transforms

logger = logging.getLogger("ActiveLearning")


class CustomImageNet(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        self.debug_mode = kwargs.pop('debug_mode', False)
        self.num_classes = 1000
        super().__init__(*args, **kwargs)

    def __len__(self):
        if self.debug_mode:
            return 50
        return super().__len__()

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__[i]

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, index


def get_data_imagenet(data_path, debug_mode=False):
    train_transform, val_transform = get_transforms()

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    train_set = CustomImageNet(traindir, transform=train_transform, debug_mode=debug_mode)
    val_set = CustomImageNet(valdir, transform=val_transform, debug_mode=debug_mode)

    # The AL set is the same data as the train set, without the input
    # transformations (random cropping, flipping, etc..)
    al_set = CustomImageNet(traindir, transform=val_transform, debug_mode=debug_mode)

    return train_set, val_set, al_set


def get_transforms():
    logger.info("(Data loading) No crops in validation")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), normalize, ])
    val_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize, ])
    return input_transform, val_transform
