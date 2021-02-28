# ------------------------------------------------------------------------------
# VGG-19 architecture
# ------------------------------------------------------------------------------

import torch.nn as nn

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # Normalize the input for VGG
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), requires_grad=False)  # Normalization means for training set of VGG
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), requires_grad=False)   # Normalization stds for training set of VGG

        # VGG features
        # BLOCK 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # BLOCK 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # BLOCK 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_4 = nn.ReLU(inplace=True)
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # BLOCK 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_4 = nn.ReLU(inplace=True)
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # BLOCK 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_4 = nn.ReLU(inplace=True)

    def forward(self, x, layer_list):
        out = {}

        x = (x - self.mean) / self.std  # Normalize the input

        out['relu1_1'] = self.relu1_1(self.conv1_1(x))
        out['relu1_2'] = self.relu1_2(self.conv1_2(out['relu1_1']))

        out['relu2_1'] = self.relu2_1(self.conv2_1(self.mpool1(out['relu1_2'])))
        out['relu2_2'] = self.relu2_2(self.conv2_2(out['relu2_1']))

        out['relu3_1'] = self.relu3_1(self.conv3_1(self.mpool2(out['relu2_2'])))
        out['relu3_2'] = self.relu3_2(self.conv3_2(out['relu3_1']))
        out['relu3_3'] = self.relu3_3(self.conv3_3(out['relu3_2']))
        out['relu3_4'] = self.relu3_4(self.conv3_4(out['relu3_3']))

        out['relu4_1'] = self.relu4_1(self.conv4_1(self.mpool3(out['relu3_4'])))
        out['relu4_2'] = self.relu4_2(self.conv4_2(out['relu4_1']))
        out['relu4_3'] = self.relu4_3(self.conv4_3(out['relu4_2']))
        out['relu4_4'] = self.relu4_4(self.conv4_4(out['relu4_3']))

        out['relu5_1'] = self.relu5_1(self.conv5_1(self.mpool4(out['relu4_4'])))
        out['relu5_2'] = self.relu5_2(self.conv5_2(out['relu5_1']))
        out['relu5_3'] = self.relu5_3(self.conv5_3(out['relu5_2']))
        out['relu5_4'] = self.relu5_4(self.conv5_4(out['relu5_3']))

        return [out[key] for key in layer_list]
