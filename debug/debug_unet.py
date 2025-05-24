import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch) # Input to DoubleConv3D is from the previous stage
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock3D(nn.Module):
    """
    An upsampling block for 3D U-Net.
    It takes features from the lower resolution and a skip connection.
    """
    def __init__(self, in_ch_up, in_ch_skip, out_ch, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample

        if use_upsample:
            # nn.Upsample does not change channels, so we need a 1x1 conv to reduce them
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(in_ch_up, in_ch_up // 2, kernel_size=1)
            )
            # The input channels to the final DoubleConv3D will be (in_ch_up // 2) + in_ch_skip
            conv_in_channels = (in_ch_up // 2) + in_ch_skip
        else:
            # ConvTranspose3d directly reduces channels and upsamples
            self.upsample = nn.ConvTranspose3d(in_ch_up, in_ch_up // 2, kernel_size=2, stride=2)
            # The input channels to the final DoubleConv3D will be (in_ch_up // 2) + in_ch_skip
            conv_in_channels = (in_ch_up // 2) + in_ch_skip

        self.conv = DoubleConv3D(conv_in_channels, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)

        # Pad if necessary due to odd dimensions after upsampling
        if x.shape[2:] != skip.shape[2:]:
            diff_D = skip.shape[2] - x.shape[2]
            diff_H = skip.shape[3] - x.shape[3]
            diff_W = skip.shape[4] - x.shape[4]

            padding = [
                diff_W // 2, diff_W - diff_W // 2,  # Width padding
                diff_H // 2, diff_H - diff_H // 2,  # Height padding
                diff_D // 2, diff_D - diff_D // 2   # Depth padding
            ]
            x = F.pad(x, padding)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, use_upsample_in_upblock=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.use_upsample_in_upblock = use_upsample_in_upblock

        # Encoding Path
        self.enc1 = DoubleConv3D(in_channels, base_filters)      # Input: 1, Output: 64
        self.pool1 = nn.MaxPool3d(2) # Output: 64

        self.enc2 = DoubleConv3D(base_filters, base_filters * 2) # Input: 64, Output: 128
        self.pool2 = nn.MaxPool3d(2) # Output: 128

        self.enc3 = DoubleConv3D(base_filters * 2, base_filters * 4) # Input: 128, Output: 256
        self.pool3 = nn.MaxPool3d(2) # Output: 256

        # Bottleneck
        self.bottleneck = DoubleConv3D(base_filters * 4, base_filters * 8) # Input: 256, Output: 512

        # Decoding Path
        # UpBlock3D(in_ch_up, in_ch_skip, out_ch)

        # U-Net's decoding path usually reverses the encoding path's channel progression.
        # d3 should output base_filters * 4 to match enc3's input filters (before double conv)
        self.up3 = UpBlock3D(base_filters * 8, base_filters * 4, base_filters * 4, use_upsample=self.use_upsample_in_upblock)
        # in_ch_up=bottleneck_output (512), in_ch_skip=enc3_output (256), out_ch=desired_output_of_this_stage (256)

        self.up2 = UpBlock3D(base_filters * 4, base_filters * 2, base_filters * 2, use_upsample=self.use_upsample_in_upblock)
        # in_ch_up=up3_output (256), in_ch_skip=enc2_output (128), out_ch=desired_output_of_this_stage (128)

        self.up1 = UpBlock3D(base_filters * 2, base_filters, base_filters, use_upsample=self.use_upsample_in_upblock)
        # in_ch_up=up2_output (128), in_ch_skip=enc1_output (64), out_ch=desired_output_of_this_stage (64)

        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1) # Input: 64, Output: 1 (or N)

    def forward(self, x):
        # Encoding Path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3) # This was missing in your original forward pass after enc3

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoding Path
        # up_block(x_from_lower_res_up, skip_connection)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        output = self.out_conv(d1)
        return output

# Example Usage:
if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 64, 64, 64) # Batch, Channels, Depth, Height, Width

    model = UNet3D(in_channels=1, out_channels=1, base_filters=16) # Using 16 for smaller test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    print(f"Input shape: {input_tensor.shape}")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    assert output.shape == input_tensor.shape, "Output shape does not match input shape!"
    print("U-Net 3D model test successful!")

    # Test with a different output channel for multi-class segmentation
    model_multiclass = UNet3D(in_channels=1, out_channels=3, base_filters=16)
    model_multiclass.to(device)
    output_multiclass = model_multiclass(input_tensor)
    print(f"Output shape for multi-class: {output_multiclass.shape}")
    assert output_multiclass.shape[0] == input_tensor.shape[0] and \
           output_multiclass.shape[2:] == input_tensor.shape[2:], \
           "Multi-class output shape mismatch!"
    print("Multi-class U-Net 3D model test successful!")

    # Test with ConvTranspose3d for upsampling
    model_conv_transpose = UNet3D(in_channels=1, out_channels=1, base_filters=16, use_upsample_in_upblock=False)
    model_conv_transpose.to(device)
    output_conv_transpose = model_conv_transpose(input_tensor)
    print(f"Output shape with ConvTranspose3d: {output_conv_transpose.shape}")
    assert output_conv_transpose.shape == input_tensor.shape, \
           "Output shape with ConvTranspose3d does not match input shape!"
    print("U-Net 3D model with ConvTranspose3d test successful!")