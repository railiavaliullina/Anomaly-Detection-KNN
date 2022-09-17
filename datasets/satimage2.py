from torch.utils.data import Dataset
from pathlib import Path
import pickle
import torch.nn.functional as f
import torch

import config


class Satimage2Dataset(Dataset):
    r"""
    https://odds.cs.stonybrook.edu/satimage-2-dataset/
    """
    def __init__(self, data_path: Path = config.DATA_PATH):

        with open(data_path / 'satimage2', 'rb') as data_file:
            data = pickle.load(data_file)
            self.vectors, self.labels = data['vectors'], data['labels'].reshape(-1)
        # self.vectors = f.normalize(torch.tensor(self.vectors), p=2)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]
