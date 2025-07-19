import torch
import torch.nn as nn
import torch.nn.functional as F

# 10 epochs - 80.49%;
# 20 epochs - 84.4%; 
# 50 epochs - 88.4%; 
# 80 epochs - 89.8%; 
# 100 epochs - 90.4%

class HeheNet(nn.Module):
    def __init__(self):
        """
        Initialize the HeheNet convolutional neural network architecture with four convolutional layers, batch normalization, max pooling, dropout, and two fully connected layers for classification.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))              
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   
        x = x.view(x.size(0), -1)                        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



