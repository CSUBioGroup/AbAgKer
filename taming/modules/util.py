import torch
import torch.nn as nn


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

