import torch.nn as nn
import torch.nn.functional as F

'''
define networks
'''
class Net(nn.Module):
    def __init__(self, num_in_channel, num_filter):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channel, num_filter, 3, padding=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.conv3 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.conv4 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 40x40
        x = self.pool(F.relu(self.conv2(x))) # 20x20
        x = self.pool(F.relu(self.conv3(x))) # 10x10
        x = self.pool(F.relu(self.conv4(x))) # 5x5
        x = x.view(x.size()[0], -1) # 3200
        return x