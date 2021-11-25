import logging

import numpy as np
from torchvision import datasets, transforms

from .custom_cifar10 import CustomCIFAR10

logger = logging.getLogger("ActiveLearning")


class ImbalanceCifar10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        self.debug_mode = kwargs.pop('debug_mode', False)
        self.num_classes = 10

        imbalance_args = kwargs.pop('imbalance_args', None)
        self.imbalance_type = imbalance_args['imbalance_type']
        self.imbalance_factor = imbalance_args['imbalance_factor']
        self.imbalance_seed = imbalance_args['imbalance_seed']

        super().__init__(*args, **kwargs)

        np.random.seed(self.imbalance_seed)
        if self.imbalance_type in ["exp", "step", "exp-uniform", "step-uniform", "minor-equal"]:
            self.img_num_list = self.get_img_num_per_cls(self.num_classes, self.imbalance_type,
                                                         self.imbalance_factor)
            self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, num_classes, imbalance_type, imbalance_factor):
        img_max = len(self.data) / num_classes
        img_num_per_cls = []
        if imbalance_type == "exp":
            for cls_idx in range(num_classes):
                num = img_max * (imbalance_factor ** (cls_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif imbalance_type == "step":
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max * imbalance_factor))
        else:
            raise ValueError("Choose a valid imbalance_type: one of exp or step. ")
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_num_classes_list(self):
        if self.imbalance_type == 'none':
            return [5000] * 10
        num_classes_list = []
        for i in range(self.num_classes):
            num_classes_list.append(self.num_per_cls_dict[i])
        return num_classes_list

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


def get_data_imbalanced_cifar10(data_path, input_size=(32, 32), supervised=False, debug_mode=False,
                               imbalance_args=None):
    train_transform, val_transform = get_transforms(input_size, supervised)

    train_set = ImbalanceCifar10(data_path, train=True, transform=train_transform, download=True,
                                 imbalance_args=imbalance_args, debug_mode=debug_mode, )
    val_set = CustomCIFAR10(data_path, train=False, transform=val_transform, download=True,
                            debug_mode=debug_mode)

    # The AL set is the same data as the train set, without the input
    # transformations (random cropping, flipping, etc..)
    al_set = ImbalanceCifar10(data_path, train=True, transform=val_transform,
                              imbalance_args=imbalance_args, debug_mode=debug_mode, )

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
