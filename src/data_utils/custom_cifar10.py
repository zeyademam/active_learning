import logging

from torchvision import datasets, transforms

logger = logging.getLogger("ActiveLearning")


class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        self.debug_mode = kwargs.pop('debug_mode', False)
        self.num_classes = 10
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


def get_data_cifar10(data_path, input_size=(32, 32), supervised=False, debug_mode=False):
    train_transform, val_transform = get_transforms(input_size, supervised)

    train_set = CustomCIFAR10(data_path, train=True, transform=train_transform, download=True,
                              debug_mode=debug_mode, )
    val_set = CustomCIFAR10(data_path, train=False, transform=val_transform, download=True,
                            debug_mode=debug_mode, )

    # The AL set is the same data as the train set, without the input
    # transformations (random cropping, flipping, etc..)
    al_set = CustomCIFAR10(data_path, train=True, transform=val_transform, debug_mode=debug_mode, )

    return train_set, val_set, al_set


def get_transforms(input_size, supervised):
    logger.info("(Data loading) Random crops of " + str(input_size) + " in training")
    logger.info("(Data loading) No crops in validation")

    input_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010)), ])
    return input_transform, val_transform
