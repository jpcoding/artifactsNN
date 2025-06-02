import torch
import torch.nn as nn
import torch.nn.functional as F

# class ARCNN(nn.Module):
#     def __init__(self):
#         super(ARCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding='same')
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=7, padding='same')
#         self.conv3 = nn.Conv2d(32, 16, kernel_size=1, padding='same')
#         self.conv4 = nn.Conv2d(16, 1, kernel_size=5, padding='same')
#         self._initialize_weights()

#     def forward(self, x):
#         if self.conv1.weight.dtype != x.dtype:
#             self.to(dtype=x.dtype, device=x.device)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.conv4(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, mean=0.0, std=0.001)
#                 nn.init.constant_(m.bias, 0)

# class ARCNN(nn.Module):
#     def __init__(self, input_channels=1, num_filters=64):
#         super(ARCNN, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(num_filters, 1, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = self.relu1(self.conv1(x))
#         x = self.relu2(self.conv2(x))
#         x = self.relu3(self.conv3(x))
#         x = self.conv4(x)
#         return x
    

class ARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=7)
        self.conv22 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=5)

        self.pad2 = nn.ReplicationPad2d(2)
        self.pad3 = nn.ReplicationPad2d(3)
        self.pad4 = nn.ReplicationPad2d(4)

        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.pad4(x)
        x = self.relu(self.conv1(x))

        x = self.pad3(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv22(x))

        x = self.pad2(x)
        x = self.conv3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

import torch
import torch.nn as nn

class ARCNNResidual(nn.Module):
    def __init__(self, in_channels=2):  # <-- default to 2 for residual + e_scalar input
        """
        ARCNNResidual Model for learning compression residuals with optional scalar input.
        
        Args:
            in_channels (int): Number of input channels. 
                               Use 1 for plain input, 2 for (input + scalar map).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=7)
        self.conv22 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=5)

        self.pad2 = nn.ReplicationPad2d(2)
        self.pad3 = nn.ReplicationPad2d(3)
        self.pad4 = nn.ReplicationPad2d(4)

        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()
        
    def forward(self, x, e_scalar):
        """
        Args:
            x (Tensor): Compressed input of shape [B, 1, H, W]
            e_scalar (Tensor): Batch of scalar error ranges, shape [B]
        Returns:
            predicted_residual (Tensor): Output residual map of shape [B, 1, H, W]
        """
        B, C, H, W = x.shape
        e_map = e_scalar.view(B, 1, 1, 1).expand(-1, 1, H, W)  # Broadcast [B, 1, H, W]
        x = torch.cat([x, e_map], dim=1)  # → [B, 2, H, W]

        x = self.pad4(x)
        x = self.relu(self.conv1(x))
        x = self.pad3(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv22(x))
        x = self.pad2(x)
        x = self.conv3(x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
