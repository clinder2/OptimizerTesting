import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from shampoo.shampoo import Shampoo

class MatrixSimple(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A=A

    def forward(self, X):
        return 2*(X-self.A)