import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Still need to add some more stuff I think
class Meta_Model(nn.Module):
    def __init__(self, dropout = 0.5, time_step_back = 2):
        super(Meta_Model, self).__init__()

        # We wish to include basin, sub_basin, pressures, lat, lon, category, season
        in_channels = time_step_back * 3 + 9
        self.fc1 = nn.Linear(in_features=in_channels, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=3)

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

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = self.fc2(x)
        return x
