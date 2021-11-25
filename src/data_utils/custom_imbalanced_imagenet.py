"""see https://arxiv.org/pdf/1904.05160v2.pdf

"""
import logging
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .custom_imagenet import CustomImageNet

logger = logging.getLogger("ActiveLearning")
cur_dir = os.path.dirname(os.path.abspath(__file__))


class ImbalanceImagenet(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels
        self.num_classes = 1000

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__[i]

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index


def get_transforms():
    logger.info("(Data loading) No crops in validation")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), normalize, ])
    val_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize, ])
    return input_transform, val_transform


def get_data_imbalanced_imagenet(data_path):
    train_transform, val_transform = get_transforms()

    traindir = os.path.join(data_path)
    valdir = os.path.join(data_path, 'val')

    train_set = ImbalanceImagenet(root=traindir,
                                  txt=os.path.join(cur_dir, "ImageNet_LT", "ImageNet_LT_train.txt"),
                                  transform=train_transform)
    val_set = CustomImageNet(valdir, transform=val_transform)

    # The AL set is the same data as the train set, without the input
    # transformations (random cropping, flipping, etc..)
    al_set = ImbalanceImagenet(root=traindir,
                               txt=os.path.join(cur_dir, "ImageNet_LT", "ImageNet_LT_train.txt"),
                               transform=val_transform)
    return train_set, val_set, al_set


if __name__ == "__main__":
    txt = os.path.join(os.dirname(__file__), "ImageNet_LT", "val")
