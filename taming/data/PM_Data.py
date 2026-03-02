import os
import numpy as np
from torch.utils.data import Dataset
import json
import torch
import random

class CustomBase(Dataset):
    def __init__(self, *args, data_seed=None, **kwargs):
        super().__init__() 
        self.data = []
        if data_seed is not None:
            self.data_seed = data_seed
        else:
            try:
                import pytorch_lightning as pl
                import os
                pl_seed = os.environ.get('PL_GLOBAL_SEED')
                if pl_seed is not None:
                    self.data_seed = int(pl_seed)
                else:
                    self.data_seed = 23  # init
            except:
                self.data_seed = 23  # init

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class DataTrain(CustomBase):
    def __init__(self, training_list_file, data_seed=None):
        super().__init__(data_seed=data_seed)

        # using a constant seed to ensure result consistence
        random.seed(self.data_seed)
        np.random.seed(self.data_seed)
        with open(training_list_file, "r") as f:
            self.data = json.load(f)
        random.shuffle(self.data)
        
class DataVal(CustomBase):
    def __init__(self, val_list_file, data_seed=None):
        super().__init__(data_seed=data_seed)

        random.seed(self.data_seed)
        np.random.seed(self.data_seed)
        with open(val_list_file, "r") as f:
            self.data = json.load(f)
        random.shuffle(self.data)
