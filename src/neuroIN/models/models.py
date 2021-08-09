import torch.nn as nn
from torch import cat, flatten

class CNN2D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2, num_channels=64, temp_kernel_length=128, spac_kernel_length=16, F1=8, D=2, F2=16, size_for_fc=32):
        '''
        size_for_fc should be automated
        '''

        super(CNN2D, self).__init__()

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

        self.output = nn.Sequential(
            nn.Linear(size_for_fc, self.n_classes)
            )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size()[0], -1)
        x = self.output(x)
        return x

class MultiBranchCNN2D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2, num_channels=64, temp_kernel_lengths=[64, 128], spac_kernel_lengths=[8, 16], F1=8, D=2, F2=16):
        assert len(temp_kernel_lengths) == len(spac_kernel_lengths), 'Must have equal number of Temporal and Spatial lengths'

        super(MultiBranchCNN2D, self).__init__()

        self.dropout_p = dropout_p
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.temp_kernel_lengths = temp_kernel_lengths
        self.spac_kernel_lengths = spac_kernel_lengths

        self.F1 = F1
        self.D = D
        self.F2 = F2


        self.branches = []
        for (temp_kernel_length, spac_kernel_length) in zip(temp_kernel_lengths, spac_kernel_lengths):
            conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, temp_kernel_length)),
            nn.BatchNorm2d(self.F1)
            )

            conv2 = nn.Sequential(
            nn.Conv2d(self.F1, (self.F1 * self.D), (self.num_channels, 1)),
            nn.BatchNorm2d((self.F1 * self.D)),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=self.dropout_p)
            )

            conv3 = nn.Sequential(
            nn.Conv2d((self.F1 * self.D), (self.F1 * self.D), (1, spac_kernel_length)),
            nn.Conv2d((self.F1 * self.D), self.F2, 1),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=self.dropout_p)
            )

            self.branches.append(nn.Sequential(conv1, conv2, conv3))

        self.output = nn.Sequential(
            nn.Linear(496, self.n_classes)
            )

    def forward(self, x):
        xs = [branch(x) for branch in self.branches]
        xs = [x.view(x.size()[0], -1) for x in xs]
        x = cat(xs, 1)
        x = self.output(x)
        return x


class CNN3D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2):
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

        self.output = nn.Sequential(
            nn.Linear(9984, self.n_classes)
            )

    def forward(self, x):
        x = self.layers(x)
        x = flatten(x, start_dim=1)
        x = self.output(x)
        return x