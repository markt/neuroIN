import torch.nn as nn
from torch import cat, flatten, unsqueeze, zeros
# from torch.nn.modules.activation import ReLU

class MultiBranchCNN2D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2, num_channels=64, temp_kernel_lengths=[64, 128], spac_kernel_lengths=[8, 16], F1=8, D=2, F2=16, fc_size=32, **extras):
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


        self.conv_branches = []
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

            self.conv_branches.append(nn.Sequential(conv1, conv2, conv3))

        self.fc_branches = []
        for conv_branch in self.conv_branches:
            dummy = zeros((1, 1, 64, 640))
            fc_input_size = flatten(conv_branch(dummy)).shape[0]

            fc1 = nn.Sequential(
                nn.Linear(fc_input_size, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.ReLU()
            )
            fc2 = nn.Sequential(
                nn.Linear(fc_size, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.ReLU()
            )
            fc_output = nn.Sequential(
                nn.Linear(fc_size, n_classes),
                nn.Softmax(dim=1)
            )

            self.fc_branches.append(nn.Sequential(fc1, fc2, fc_output))
        
        output_input_size = len(self.fc_branches) * n_classes
        self.output = nn.Sequential(
            nn.Linear(output_input_size, self.n_classes)
            )

    def forward(self, x):
        x = unsqueeze(x, 1)
        xs = [branch(x) for branch in self.conv_branches]
        xs = [x.view(x.size()[0], -1) for x in xs]
        xs = [branch(x) for (x, branch) in zip(xs, self.fc_branches)]
        x = cat(xs, 1)
        x = self.output(x)
        return x


class MultiBranchCNN3D(nn.Module):
    '''
    Subclass of PyTorch nn.Module for a basic 2D CNN
    '''

    def __init__(self, dropout_p=0.5, n_classes=2, fc_size=32, **extras):
        super(MultiBranchCNN2D, self).__init__()

        self.dropout_p = dropout_p
        self.n_classes = n_classes

        self.common_conv = nn.Sequential(
            nn.Conv3d(1, 16, (3, 3, 5), stride=(2, 2, 4), padding=(2, 2, 0), dilation=1),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p)
            )
        
        self.conv_branches = []
        for k_temp in [1, 3, 5]:
            conv1 = nn.Sequential(
                nn.Conv3d(16, 32, (2, 2, k_temp), stride=(2, 2, 2), padding=(1, 1, 0), dilation=1),
                nn.BatchNorm3d(32),
                nn.ELU(),
                nn.Dropout(p=self.dropout_p)
                )
            
            conv2 = nn.Sequential(
                nn.Conv3d(32, 64, (3, 3, k_temp), stride=(2, 2, 2), padding=(1, 1, 0), dilation=1),
                nn.BatchNorm3d(64),
                nn.ELU(),
                nn.Dropout(p=self.dropout_p)
                )
                
            self.conv_branches.append(nn.Sequential(conv1, conv2))

        self.fc_branches = []
        for conv_branch in self.conv_branches:
            dummy = zeros((1, 1, 5, 5, 640))
            fc_input_size = flatten(conv_branch(dummy)).shape[0]

            fc1 = nn.Sequential(
                nn.Linear(fc_input_size, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.ReLU()
            )
            fc2 = nn.Sequential(
                nn.Linear(fc_size, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.ReLU()
            )
            fc_output = nn.Sequential(
                nn.Linear(fc_size, n_classes),
                nn.Softmax(dim=1)
            )

            self.fc_branches.append(nn.Sequential(fc1, fc2, fc_output))
        
        output_input_size = len(self.fc_branches) * n_classes
        self.output = nn.Sequential(
            nn.Linear(output_input_size, self.n_classes)
            )
            

    def forward(self, x):
        x = unsqueeze(x, 1)
        xs = [branch(x) for branch in self.conv_branches]
        xs = [x.view(x.size()[0], -1) for x in xs]
        xs = [branch(x) for (x, branch) in zip(xs, self.fc_branches)]
        x = cat(xs, 1)
        x = self.output(x)
        return x