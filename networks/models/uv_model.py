import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
# Need to modify
class UV_Model(nn.Module):
    def __init__(self, time_steps_back=1, pressure_levels=5, feature_pred=False):
        super(UV_Model, self).__init__()

        self.in_channels = 2*(time_steps_back+1)*pressure_levels
        self.feature_pred = feature_pred
        
        self.dropout = nn.Dropout(0.5)

        # Out: 32x157x157
        # self.conv0 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=4, stride=1, padding=0, groups=1, bias=True)
        # self.conv0_bn = nn.BatchNorm2d(64)

        # Out: 96x65x65
        self.conv0 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=2, padding=0, groups=1, bias=True)
        self.conv0_bn = nn.BatchNorm2d(96)

        # Pool with 3x3 max 2 stride
        # Out: 32x32x96

        # Out: 64x78x78
        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        # self.conv1_bn = nn.BatchNorm2d(64)

        # Out 32x32x256
        self.conv1 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(256)

        # Pool with 3x3 max + 2 stride
        # Out: 15x15x256

        # Out: 128x76x76
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        # self.conv2_bn = nn.BatchNorm2d(128)

        # Out: 15x15x384
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(384)

        # Apply maxpool 2d 128x38x38

        # Out: 256x18x18
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        # self.conv3_bn = nn.BatchNorm2d(256) # We can play with this

        # Out: 15x15x384
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv3_bn = nn.BatchNorm2d(384)

        # Out: 8x8x256
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        self.conv4_bn = nn.BatchNorm2d(256)

        # MAx pool with 3x3 s=2, p=1
        # Out: 5x5x256

        # Out: 512x8x8
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        # self.conv4_bn = nn.BatchNorm2d(256) # We can play with this

        # Apply maxpool 2d 2x1

        self.fc1 = nn.Linear(in_features=256*4*4, out_features=2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc2_bn = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=3)
        self.init_weights()

    def init_weights(self):
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d) and idx==1:
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.bias,mean=0, std=1)
            if isinstance(m, nn.Conv2d) and idx!=1:
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                nn.init.normal_(m.bias,mean=0, std=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and idx==1:
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.bias,mean=0, std=1)
            elif isinstance(m, nn.Linear) and idx!=1:
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                nn.init.normal_(m.bias,mean=0, std=1)

    def forward(self, example):        
        x = example
        x = F.relu(self.conv0_bn(self.conv0(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        # print(f"View after conv0 {x.size()}")
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        # print(f"View after conv1 {x.size()}")
        x = F.relu(self.conv2_bn(self.conv2(x)))
        # print(f"View after conv2 {x.size()}")
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        # print(f"View after conv3 {x.size()}")
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # print(x.size())
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))

        if self.feature_pred == True:
            return self.fc2(x)

        x = self.dropout(x)
        x = self.fc3(x)
        # print(x.size())
        return x
