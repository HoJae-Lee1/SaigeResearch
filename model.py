'''
    This code is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This code is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this code.  If not, see <https://www.gnu.org/licenses/>.

    Copyright (c) Saige Research
    All rights reserved.
'''

import torch
import torch.nn as nn
import torchvision.models as models


class sv_network(nn.Module):
    def __init__(self):
        super().__init__()

        resnet_model = models.resnet18(pretrained=True)

        self.sv_layer0 = resnet_model.conv1
        self.sv_layer1 = resnet_model.bn1
        self.sv_layer2 = resnet_model.relu
        self.sv_layer3 = resnet_model.maxpool

        self.sv_layer4 = resnet_model.layer1[0]
        self.sv_layer5 = resnet_model.layer1[1]

        self.sv_layer6 = resnet_model.layer2[0]
        self.sv_layer7 = resnet_model.layer2[1]

        self.sv_layer8 = resnet_model.layer3[0]
        self.sv_layer9 = resnet_model.layer3[1]

        self.sv_layer10 = resnet_model.layer4[0]
        self.sv_layer11 = resnet_model.layer4[1]

        self.sv_layer12 = resnet_model.avgpool
        self.sv_layer13 = resnet_model.fc

    def forward(self, x):
        x = self.sv_layer0(x)
        x = self.sv_layer1(x)
        x = self.sv_layer2(x)
        x = self.sv_layer3(x)

        x = self.sv_layer4(x)
        x = self.sv_layer5(x)

        x = self.sv_layer6(x)
        x = self.sv_layer7(x)

        x = self.sv_layer8(x)
        x = self.sv_layer9(x)

        x = self.sv_layer10(x)
        x = self.sv_layer11(x)

        x = self.sv_layer12(x)
        x = torch.flatten(x, 1)
        x = self.sv_layer13(x)

        return x
