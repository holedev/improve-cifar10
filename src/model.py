import torch
import torch.nn as nn
import torch.nn.functional as F

# 5 epoch - 58%
# class HeheNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, 2)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(16*7*7, 512)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# 20 epoch - 76%; 50 epoch - 80.5%; 64 epoch - 82.2%
# class HeheNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
        
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
        
#         self.fc1 = nn.Linear(64*16*16, 512)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = x.view(x.size(0), -1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

# 20 epoch - 81.1%; 50 epoch - 86.4%; 64 epoch - 87.0%
# class HeheNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
        
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)

#         # Sau 2 lần MaxPool: từ 32x32 → 16x16 → 8x8
#         self.fc1 = nn.Linear(128 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))    # [B, 32, 32, 32]
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 16, 16]
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 128, 8, 8]
#         x = x.view(x.size(0), -1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

# 20 epoch - 84.4%; 50 epoch - 88.4%; 80 epoch - 89.8%; 100 epoch - 90.4%
class HeheNet(nn.Module):
    def __init__(self):
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



