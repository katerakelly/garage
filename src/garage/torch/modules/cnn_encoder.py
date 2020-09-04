import torch

from torch import nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Implements a CNN encoder for processing image observations
    Using the architecture from MELD
    """
    def __init__(self,
                in_channels,
                output_dim):
        super().__init__()
        # output is batch x feature x 1 x 1 for inputs of 64x64
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.output_dim = output_dim


    def forward(self, in_):
        in_ = in_.view(-1, 64, 64, 4)
        in_ = in_.permute(0, 3, 1, 2).contiguous()
        in_ = F.relu(self.conv1(in_))
        in_ = F.relu(self.conv2(in_))
        in_ = F.relu(self.conv3(in_))
        in_ = F.relu(self.conv4(in_))
        in_ = F.relu(self.conv5(in_)).squeeze()
        return in_
