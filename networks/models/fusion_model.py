import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Need to modify...
# Please see https://github.com/sophiegif/FusionCNN_hurricanes
class Fusion_Model(nn.Module):
    def __init__(self, time_steps_back=1, pressure_levels=5, feature_pred = False):
        super(Fusion_Model, self).__init__()
        self.in_channels = 2*(time_steps_back+1)*pressure_levels
        self.feature_pred = feature_pred

        self.dropout = nn.Dropout(0.25)

        self.conv0_uv = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=11, stride=2, padding=0, groups=1, bias=True)
        self.conv0_bn_uv = nn.BatchNorm2d(96)

        self.conv1_uv = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv1_bn_uv = nn.BatchNorm2d(256)

        self.conv2_uv = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv2_bn_uv = nn.BatchNorm2d(384)

        self.conv3_uv = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv3_bn_uv = nn.BatchNorm2d(384)

        self.conv4_uv = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        self.conv4_bn_uv = nn.BatchNorm2d(256)

        self.fc1_uv = nn.Linear(in_features=256*4*4, out_features=2048)
        self.fc1_bn_uv = nn.BatchNorm1d(2048)

        self.fc2_uv = nn.Linear(in_features=2048, out_features=2048)
        self.fc2_bn_uv = nn.BatchNorm1d(2048)

        self.conv0_z = nn.Conv2d(in_channels=10, out_channels=96, kernel_size=11, stride=2, padding=0, groups=1, bias=True)
        self.conv0_bn_z = nn.BatchNorm2d(96)

        self.conv1_z = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv1_bn_z = nn.BatchNorm2d(256)

        self.conv2_z = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv2_bn_z = nn.BatchNorm2d(384)

        self.conv3_z = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv3_bn_z = nn.BatchNorm2d(384)

        self.conv4_z = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=0, groups=1, bias=True)
        self.conv4_bn_z = nn.BatchNorm2d(256)

        self.fc1_z = nn.Linear(in_features=256*4*4, out_features=2048)
        self.fc1_bn_z = nn.BatchNorm1d(2048)

        self.fc2_z = nn.Linear(in_features=2048, out_features=2048)
        self.fc2_bn_z = nn.BatchNorm1d(2048)

        self.fc1 = nn.Linear(in_features=(2048*2 + (time_steps_back+1) * 3 + 9), out_features=2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=128)
        self.fc3_bn = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(in_features=128, out_features=3)

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
    
    def forward(self,example_both):
        example = example_both[0]
        meta_example = example_both[1]

        list1 = [*list(range(0,10)), *list(range(15,25))]
        list2 = [*list(range(10,15)), *list(range(25,30))]
        x1 = example[:,list1,:,:]
        x2 = example[:,list2,:,:]

        x1 = F.relu(self.conv0_bn_uv(self.conv0_uv(x1)))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=0)
        x1 = F.relu(self.conv1_bn_uv(self.conv1_uv(x1)))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=0)
        x1 = F.relu(self.conv2_bn_uv(self.conv2_uv(x1)))
        x1 = F.relu(self.conv3_bn_uv(self.conv3_uv(x1)))
        x1 = F.relu(self.conv4_bn_uv(self.conv4_uv(x1)))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=1)
        x1 = x1.view(-1, 256*4*4)
        x1_skip = self.fc1_uv(x1)
        x1 = F.relu(self.fc1_bn_uv(self.fc1_uv(x1)))
        x1 = F.relu(self.fc2_bn_uv(self.fc2_uv(x1)))

        x2 = F.relu(self.conv0_bn_z(self.conv0_z(x2)))
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2, padding=0)
        x2 = F.relu(self.conv1_bn_z(self.conv1_z(x2)))
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2, padding=0)
        x2 = F.relu(self.conv2_bn_z(self.conv2_z(x2)))
        x2 = F.relu(self.conv3_bn_z(self.conv3_z(x2)))
        x2 = F.relu(self.conv4_bn_z(self.conv4_z(x2)))
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2, padding=1)
        x2 = x2.view(-1, 256*4*4)
        x2_skip = self.fc1_z(x2)
        x2 = F.relu(self.fc1_bn_z(self.fc1_z(x2)))
        x2 = F.relu(self.fc2_bn_z(self.fc2_z(x2)))

        if self.feature_pred:
            feature_pred_vec = torch.cat((x1_skip,x2_skip, meta_example.float()), dim=1)
            return feature_pred_vec

        x = torch.cat((x1,x2, meta_example.float()), dim=1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.fc4(x)
        return x