import torch.nn as nn
from torch import flatten, unsqueeze, zeros

class CNN2Plus1D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2, l2=32, shape=(5, 5, 640), **extras):
        """init for CNN2Plus1D

        :param dropout_p: [description], defaults to 0.5
        :type dropout_p: float, optional
        :param n_classes: [description], defaults to 2
        :type n_classes: int, optional
        :param l2: the number of units in the final FC layer before classification, defaults to 32
        :type l2: int, optional
        :param shape: [description], defaults to (5, 5, 640)
        :type shape: tuple, optional
        """

        super(CNN2Plus1D, self).__init__()

        self.dropout_p = dropout_p
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, (3, 3, 1), stride=(2, 2, 1), padding=(2, 2, 0), dilation=1),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, (1, 1, 5), stride=(1, 1, 4), padding=(0, 0, 0), dilation=1),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, (2, 2, 1), stride=(2, 2, 1), padding=(1, 1, 0), dilation=1),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.Conv3d(32, 32, (1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 0), dilation=1),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0), dilation=1),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, (1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 0), dilation=1),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p)
        )

        self.layers = nn.Sequential(self.conv1, self.conv2, self.conv3)

        dummy = zeros((1, 1) + shape)
        l1 = flatten(self.layers(dummy)).shape[0]

        self.output = nn.Sequential(
            nn.Linear(l1, l2),
            nn.Linear(l2, self.n_classes)
            )

    def forward(self, x):
        x = unsqueeze(x, 1)
        x = self.layers(x)
        x = flatten(x, start_dim=1)
        x = self.output(x)
        return x