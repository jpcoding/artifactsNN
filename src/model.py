import torch
import torch.nn as nn

class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        # self.base = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=9, padding=4),
        #     nn.PReLU(),
        #     nn.Conv2d(64, 32, kernel_size=7, padding=3),
        #     nn.PReLU(),
        #     nn.Conv2d(32, 16, kernel_size=1),
        #     nn.PReLU()
        # )
        self.base = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # new!
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 1, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)


    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x
    

class ARCNNResidual(nn.Module):
    def __init__(self):
        super(ARCNNResidual, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=9, padding=4),  # 2 channels: input + error bound
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 1, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x, error_bound):
        """
        x: [B, 1, H, W]  -- quantized input
        error_bound: [B] or [B, 1, H, W]  -- scalar or map
        """
        if error_bound.dim() == 1:
            B, _, H, W = x.shape
            error_bound = error_bound.view(B, 1, 1, 1).expand(-1, 1, H, W)  # broadcast scalar to map

        x_cat = torch.cat([x, error_bound], dim=1)  # [B, 2, H, W]
        out = self.base(x_cat)
        residual = self.last(out)
        return residual



class FastARCNN(nn.Module):
    def __init__(self):
        super(FastARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.ConvTranspose2d(64, 1, kernel_size=9, stride=2, padding=4, output_padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x




class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

class CASCNN(nn.Module):
    def __init__(self):
        super(CASCNN, self).__init__()

        # Stage 1
        self.a1 = ConvBlock(1, 128)
        self.a2 = ConvBlock(128, 128)

        # Stage 2
        self.down1 = nn.AvgPool2d(2)
        self.b1 = ConvBlock(128, 128)
        self.b2 = ConvBlock(128, 128)
        self.b_up = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

        # Stage 3
        self.down2 = nn.AvgPool2d(2)
        self.c1 = ConvBlock(128, 256)
        self.c2 = ConvBlock(256, 256)
        self.c_up = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        # Stage 4
        self.down3 = nn.AvgPool2d(2)
        self.d1 = ConvBlock(256, 256)
        self.d2 = ConvBlock(256, 256)
        self.d_up = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)

        # Output layers
        self.pred_a = nn.Conv2d(128 + 129, 1, kernel_size=3, padding=1)
        self.pred_b = nn.Conv2d(128 + 129, 1, kernel_size=3, padding=1)
        self.pred_c = nn.Conv2d(128 + 129, 1, kernel_size=3, padding=1)
        self.pred_d = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        a = self.a1(x)
        a = self.a2(a)

        # Stage 2
        a_ds = self.down1(a)
        b = self.b1(a_ds)
        b = self.b2(b)
        b_up = self.b_up(b)
        b_cat = torch.cat([a, b_up], dim=1)

        # Stage 3
        b_ds = self.down2(b)
        c = self.c1(b_ds)
        c = self.c2(c)
        c_up = self.c_up(c)
        c_cat = torch.cat([b, c_up], dim=1)

        # Stage 4
        c_ds = self.down3(c)
        d = self.d1(c_ds)
        d = self.d2(d)
        d_up = self.d_up(d)

        # Predict outputs at all levels (optional, multi-scale loss)
        out_a = self.pred_a(torch.cat([a, b_up], dim=1))
        out_b = self.pred_b(torch.cat([b, c_up], dim=1))
        out_c = self.pred_c(torch.cat([c, d_up], dim=1))
        out_d = self.pred_d(d)

        # Upsample lower-res outputs to match full res
        out_b_up = F.interpolate(out_b, size=out_a.shape[-2:], mode='bilinear', align_corners=False)
        out_c_up = F.interpolate(out_c, size=out_a.shape[-2:], mode='bilinear', align_corners=False)
        out_d_up = F.interpolate(out_d, size=out_a.shape[-2:], mode='bilinear', align_corners=False)

        final_out = (out_a + out_b_up + out_c_up + out_d_up) / 4.0
        return final_out


class SRCNN(nn.Module):
    def __init__(self, in_channels=1, num_filters=64):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 1, kernel_size=3, padding=1)  # output is 1-channel
        )

    def forward(self, x):
        return self.model(x)
    

class DnCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=64, depth=20):
        super(DnCNN, self).__init__()
        layers = []

        # First layer (Conv + ReLU, no BN)
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers (Conv + BN + ReLU)
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))

        # Last layer (Conv only, to reconstruct residual)
        layers.append(nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)