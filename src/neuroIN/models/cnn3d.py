import torch.nn as nn
from torch import flatten, unsqueeze, zeros

class CNN3D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2, l2=32, shape=(5, 5, 640), **extras):
        '''
        size_for_fc should be automated
        '''

        super(CNN3D, self).__init__()

        self.dropout_p = dropout_p
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, (3, 3, 5), stride=(2, 2, 4), padding=(2, 2, 0), dilation=1),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p)
            )

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, (2, 2, 3), stride=(2, 2, 2), padding=(1, 1, 0), dilation=1),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p)
            )

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 0), dilation=1),
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