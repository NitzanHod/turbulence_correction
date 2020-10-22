import torch.nn.functional as F
import torch.nn as nn
import torch


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolution, self).__init__()

        self.convolution = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = F.relu(self.batch(self.convolution(x)))

        return out


class UpConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvolution, self).__init__()

        self.up_convolution = nn.ConvTranspose3d(in_channels, out_channels,
                                                 kernel_size=2, stride=2)

    # Center crop
    def crop(self, bridge, up):
        batch_size, n_channels, depth, layer_width, layer_height = bridge.size()
        target_batch_size, target_n_channels, target_depth, target_layer_width, target_layer_height = up.size()

        xy = (layer_width - target_layer_width) // 2
        zxy = (depth - target_depth) // 2
        # Returns a smaller block which is the same size than the block in the up part
        return bridge[:, :, zxy:(zxy + target_depth), xy:(xy + target_layer_width), xy:(xy + target_layer_width)]

    def forward(self, x, bridge):
        up = self.up_convolution(x)

        # Bridge is the opposite block of the up part
        out = torch.cat((bridge, up), 1)

        return out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down of unet
        self.conv_3_32 = Convolution(3, 32)
        self.conv_32_64 = Convolution(32, 64)
        self.conv_64_64 = Convolution(64, 64)
        self.conv_64_128 = Convolution(64, 128)
        self.conv_128_128 = Convolution(128, 128)
        self.conv_128_256 = Convolution(128, 256)
        self.conv_256_256 = Convolution(256, 256)
        self.conv_256_512 = Convolution(256, 512)

        # Up of unet
        self.conv_512_512_UpConv = UpConvolution(512, 512)
        self.conv_768_256_Conv = Convolution(768, 256)
        self.conv_256_256_Conv = Convolution(256, 256)
        self.conv_256_256_UpConv = UpConvolution(256, 256)
        self.conv_384_128_Conv = Convolution(384, 128)
        self.conv_128_128_Conv = Convolution(128, 128)
        self.conv_128_128_UpConv = UpConvolution(128, 128)
        self.conv_192_64_Conv = Convolution(192, 64)
        self.conv_64_64_Conv = Convolution(64, 64)
        self.conv_64_3 = nn.Conv3d(64, 3, 1)

    def forward(self, x):
        x = self.conv_3_32(x)
        block1 = self.conv_32_64(x)
        block2 = self.pooling(block1)
        block2 = self.conv_64_64(block2)
        block2 = self.conv_64_128(block2)
        block3 = self.pooling(block2)
        block3 = self.conv_128_128(block3)
        block3 = self.conv_128_256(block3)
        block4 = self.pooling(block3)
        block4 = self.conv_256_256(block4)
        block4 = self.conv_256_512(block4)

        up1 = self.conv_512_512_UpConv(block4, block3)
        del block4, block3
        up1 = self.conv_768_256_Conv(up1)
        up1 = self.conv_256_256_Conv(up1)

        up2 = self.conv_256_256_UpConv(up1, block2)
        del block2
        up2 = self.conv_384_128_Conv(up2)
        up2 = self.conv_128_128_Conv(up2)

        up3 = self.conv_128_128_UpConv(up2, block1)
        del block1
        up3 = self.conv_192_64_Conv(up3)
        up3 = self.conv_64_64_Conv(up3)
        up3 = self.conv_64_3(up3)
        up3 = torch.mean(up3, dim=2).unsqueeze(2)
        return up3
