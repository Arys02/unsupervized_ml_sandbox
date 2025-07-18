from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

import numpy as np
from config import PROCESSED_DATA_DIR

def get_card_dataset():
    try :
        crds = np.load(PROCESSED_DATA_DIR / "card_exploration.npy")
        return CustomDataset(crds)

    except FileNotFoundError:
        raise FileNotFoundError("Please download the dataset first.")