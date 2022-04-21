import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Need to modify...
# Please see https://github.com/sophiegif/FusionCNN_hurricanes
class Fusion_Model(nn.Module):
    def __init__(self, time_steps_back=1, pressure_levels=5):
        super(Fusion_Model, self).__init__()
        self.in_channels = 2*(time_steps_back+1)*pressure_levels

        self.conv1_uv = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        self.conv1_bn_uv = nn.BatchNorm2d(64)
        self.conv2_uv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv2_bn_uv = nn.BatchNorm2d(64)
        self.conv3_uv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv3_bn_uv = nn.BatchNorm2d(64)
        self.fc1_uv = nn.Linear(in_features=64*18*18, out_features=576)
        self.fc1_bn_uv = nn.BatchNorm1d(576)
        self.fc2_uv = nn.Linear(in_features=576, out_features=128)
        self.fc2_bn_uv = nn.BatchNorm1d(128)
        self.fc3_uv = nn.Linear(in_features=128, out_features=64)
        self.fc3_bn_uv = nn.BatchNorm1d(64)
        self.fc4_uv = nn.Linear(in_features=64, out_features=8)

        self.conv1_z = nn.Conv2d(in_channels=int(self.in_channels/2), out_channels=64, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        self.conv1_bn_z = nn.BatchNorm2d(64)
        self.conv2_z = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv2_bn_z = nn.BatchNorm2d(64)
        self.conv3_z = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.conv3_bn_z = nn.BatchNorm2d(64)

        self.fc1_z = nn.Linear(in_features=64*18*18, out_features=576)
        self.fc1_bn_z = nn.BatchNorm1d(576)
        self.fc2_z = nn.Linear(in_features=576, out_features=128)
        self.fc2_bn_z = nn.BatchNorm1d(128)
        self.fc3_z = nn.Linear(in_features=128, out_features=64)
        self.fc3_bn_z = nn.BatchNorm1d(64)
        self.fc4_z = nn.Linear(in_features=64, out_features=8)

        self.meta_in_channels = (time_steps_back+1) * 3 + 9
        self.fc1_meta = nn.Linear(in_features=self.meta_in_channels, out_features=5)
        self.fc2_meta = nn.Linear(in_features=5, out_features=5)

        self.fc5 = nn.Linear(in_features=8+8+5, out_features=8*2+5)
        self.fc6 = nn.Linear(in_features=8*2+5, out_features=9)
        self.fc7 = nn.Linear(in_features=9, out_features=3)

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
    
    def forward(self,example,meta_example):
        list1 = [*list(range(0,10)), *list(range(15,25))]
        list2 = [*list(range(10,15)), *list(range(25,30))]
        x1 = example[:,list1,:,:]
        x2 = example[:,list2,:,:]


        x1 = F.relu(self.conv1_bn_uv(self.conv1_uv(x1)))
        x1 = F.relu(self.conv2_bn_uv(self.conv2_uv(x1)))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
        x1 = F.relu(self.conv3_bn_uv(self.conv3_uv(x1)))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
        x1 = x1.view(-1, 64*18*18)

        x2 = F.relu(self.conv1_bn_z(self.conv1_z(x2)))
        x2 = F.relu(self.conv2_bn_z(self.conv2_z(x2)))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
        x2 = F.relu(self.conv3_bn_z(self.conv3_z(x2)))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
        x2 = x2.view(-1, 64*18*18)

        x1 = F.relu(self.fc1_bn_uv(self.fc1_uv(x1)))
        x1 = F.relu(self.fc2_bn_uv(self.fc2_uv(x1)))
        x1 = F.relu(self.fc3_bn_uv(self.fc3_uv(x1)))
        x1 = F.relu(self.fc4_uv(x1))

        x2 = F.relu(self.fc1_bn_z(self.fc1_z(x2)))
        x2 = F.relu(self.fc2_bn_z(self.fc2_z(x2)))
        x2 = F.relu(self.fc3_bn_z(self.fc3_z(x2)))
        x2 = F.relu(self.fc4_z(x2))

        x3 = F.relu(self.fc1_meta(meta_example.float()))
        x3 = self.fc2_meta(x3)

        x = torch.cat((x1,x2, x3), dim=1)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.fc7(x)
        return x