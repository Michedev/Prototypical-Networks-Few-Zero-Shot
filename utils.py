import torch
import numpy as np

def set_all_seeds(seed: int):
    torch.random.manual_seed(seed)
    np.random.seed(seed)