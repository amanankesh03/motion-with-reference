import torch
import torch.nn as nn
from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
    
    def forward(self, x, y):
        return 1.0 - ssim(x, y)