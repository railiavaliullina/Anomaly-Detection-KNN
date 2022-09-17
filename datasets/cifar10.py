from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pathlib import Path
import numpy as np

import config


class Cifar10(Dataset):
    r"""
    https://www.cs.toronto.edu/~kriz/cifar.html
    This class is a wrapper over the default pytorch class for ease of use for the anomaly detection task.
    Parameter 'anomaly_class' is responsible for which class will be considered anomalous, while the rest are normal.
    Available classes:
                     'airplane'
                     'automobile'
                     'bird'
                     'cat'
                     'deer'
                     'dog'
                     'frog'
                     'horse'
                     'ship'
                     'truck'
    """
    DEFAULT_DATA_PATH = config.DATA_PATH / 'cifar10'

    def __init__(self, anomaly_class: str, data_path: Path = DEFAULT_DATA_PATH):
        _dataset = CIFAR10(root=str(data_path), download=True, transform=transforms.ToTensor())
        anomaly_class_idx = _dataset.class_to_idx[anomaly_class]

        ids = []
        unique_targets_counters = {k: True for k in np.unique(_dataset.targets)}
        for i, target in enumerate(_dataset.targets):
            if unique_targets_counters[target]:
                ids.append(i)
                unique_targets_counters[target] = False
            else:
                unique_targets_counters[target] = True
        ids = np.asarray(ids)

        images = np.asarray(_dataset.data)[ids]
        targets = np.asarray(_dataset.targets)[ids]
        self.vectors = images.reshape(images.shape[0], -1)
        self.labels = np.asarray(targets == anomaly_class_idx).astype(int)
        self.images = images

    def __len__(self):
        return self.labels.shape[0]

    @staticmethod
    def apply_transform(im):
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.MEAN,
                std=config.STD)
        ])
        im = transforms_(im)
        return im

    def __getitem__(self, idx):
        images = self.apply_transform(self.images[idx])
        return images, self.labels[idx]
