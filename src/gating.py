import torch
import torch.nn as nn
import torch.nn.functional as F

class Gating(nn.Module):
    def __init__(self, num_experts, capacity=1):
        super(Gating, self).__init__()
        self.capacity = capacity
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv4 = nn.Conv2d(32, capacity * 64, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(capacity * 64, capacity * 64, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(capacity * 64, capacity * 64, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(capacity * 64, capacity * 64, 3, 1, 1)

        self.fc1 = nn.Conv2d(capacity * 64, capacity**2 * 64, 1, 1, 0)
        self.fc2 = nn.Conv2d(capacity**2 * 64, capacity**2 * 64, 1, 1, 0)
        self.fc3 = nn.Conv2d(capacity**2 * 64, num_experts, 1, 1, 0)

    # [B, C, H, W]
    def forward(self, inputs):
        x = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(x))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        if self.capacity == 1:
            x = torch.tanh(x)

        x = F.avg_pool2d(x, x.size()[2:])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = F.log_softmax(x, dim=1)

        return x[:,:,0,0]
