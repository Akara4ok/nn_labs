import pandas as pd
import numpy as np
import random
from typing import Tuple
import sys

sys.path.append('config')
from settings import RANDOM_SEED

class Dataloader():
    def __init__(self, path: str) -> None:
        self.data_path = path
        self.data = pd.read_csv(self.data_path, sep=',', encoding='utf-8')
    
    def split(self, train_percent: float, val_percent: float, test_percent: float, shuffle: bool = False, random_seed: int = RANDOM_SEED) -> Tuple:
        if shuffle:
            if random_seed:
                random.Random(random_seed).shuffle(self.data)
            else:
                random.shuffle(self.data)
        
        train_idx = int(train_percent * len(self.data))
        val_idx = train_idx + int(val_percent * len(self.data))
        
        train_data = np.array(self.data[:train_idx])
        val_data = np.array(self.data[train_idx:val_idx])
        test_data = np.array(self.data[val_idx:])

        return train_data, val_data, test_data