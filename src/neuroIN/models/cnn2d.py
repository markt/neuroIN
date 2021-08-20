import torch.nn as nn
from torch import cat, flatten, unsqueeze, zeros

class CNN2D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2, num_channels=64, shape=(64, 640), temp_kernel_length=128, spac_kernel_length=16, F1=8, D=2, F2=16, l1=224, l2=32, **extras):
        '''
        size_for_fc should be automated
        '''

        super().__init__()

        self.dropout_p = dropout_p
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.temp_kernel_length = temp_kernel_length
        self.spac_kernel_length = spac_kernel_length

        self.F1 = F1
        self.D = D
        self.F2 = F2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.temp_kernel_length)),
            nn.BatchNorm2d(self.F1)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, (self.F1 * self.D), (self.num_channels, 1)),
            nn.BatchNorm2d((self.F1 * self.D)),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=self.dropout_p)
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d((self.F1 * self.D), (self.F1 * self.D), (1, self.spac_kernel_length)),
            nn.Conv2d((self.F1 * self.D), self.F2, 1),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
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
        x = x.view(x.size()[0], -1)
        x = self.output(x)
        return x