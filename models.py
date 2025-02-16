
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return F.sigmoid(logits)*300
    



class ResNetFCN(nn.Module):
    def __init__(self, input_size=224, output_scale=300.0):
        super(ResNetFCN, self).__init__()

        # Load Pretrained ResNet
        self.backbone = models.resnet34(pretrained=True)

        # Modify first layer for single-channel input (grayscale)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove fully connected layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Upsample to 126x126 instead of 224x224
        self.upsample = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=1)  # Final output layer
        )

        # Output scaling factor
        self.output_scale = output_scale

    def forward(self, x):
        x = self.backbone(x)  # Feature extraction
        x = self.upsample(x)  # Upsample to input size (126x126)
        x = torch.sigmoid(x) * self.output_scale  # Scale output to [0, 300]
        return x



import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, output_scale=300):
        super(SegNet, self).__init__()

        # Load Pretrained VGG16 Backbone (Encoder)
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.encoder = list(vgg16_bn.features.children())[:34]  # Remove FC layers

        # Modify first conv layer to accept grayscale input
        self.encoder[0] = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)

        # Convert list to nn.Sequential
        self.encoder = nn.Sequential(*self.encoder)

        # Decoder (Mirrors the encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, n_classes, kernel_size=3, padding=1),
        )

        self.output_scale = output_scale

    def forward(self, x):
        x = self.encoder(x)  # Extract features
        x = self.decoder(x)  # Decode features back to full size
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)  # Ensure output size matches input
        x = torch.sigmoid(x) * self.output_scale  # Scale output
        return x




class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.conv0_0 = DoubleConv(n_channels, 64)
        self.conv1_0 = DoubleConv(64, 128)
        self.conv2_0 = DoubleConv(128, 256)
        self.conv3_0 = DoubleConv(256, 512)
        self.conv4_0 = DoubleConv(512, 1024)

        # Decoder with dense skip connections
        self.conv0_1 = DoubleConv(64 + 128, 64)
        self.conv1_1 = DoubleConv(128 + 256, 128)
        self.conv2_1 = DoubleConv(256 + 512, 256)
        self.conv3_1 = DoubleConv(512 + 1024, 512)

        self.conv0_2 = DoubleConv(64*2 + 128, 64)
        self.conv1_2 = DoubleConv(128*2 + 256, 128)
        self.conv2_2 = DoubleConv(256*2 + 512, 256)

        self.conv0_3 = DoubleConv(64*3 + 128, 64)
        self.conv1_3 = DoubleConv(128*3 + 256, 128)

        self.conv0_4 = DoubleConv(64*4 + 128, 64)

        # Output layers
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))

        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))

        logits = self.final(x0_4)
        return torch.sigmoid(logits) * 300  # Scale output

