# Author: Brett Kinsella
# Date: 3/9/2025
# Description:
# Implementation of the Context Encoder model trained using joint reconstruction loss (L2) and adversarial loss from the paper "Context Encoders: Feature Learning by Inpainting"
#   by Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros
#   https://arxiv.org/abs/1604.07379

import torch.nn as nn

# Listed dimensions are  C X H x W
# Designed for 227x227 input images
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # input size: 3x128x128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # output size: 64x64x64

        # input size: 64x64x64
        self.conv2 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm2 = nn.BatchNorm2d(64)
        # output size: 64x32x32

        # input size: 64x32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm3 = nn.BatchNorm2d(128)
        # output size: 128x16x16

        # input size: 128x16x16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm4 = nn.BatchNorm2d(256)
        # output size: 256x8x8

        # input size: 256x8x8
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm5 = nn.BatchNorm2d(512)
        # output size: 512x4x4

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bNorm2(self.relu2(x))

        x = self.conv3(x)
        x = self.bNorm3(self.relu3(x))

        x = self.conv4(x)
        x = self.bNorm4(self.relu4(x))

        x = self.conv5(x)
        x = self.bNorm5(self.relu5(x))

        return x
    
# Bottleneck layer
class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()
        pass
    
    def forward(self, x):
        pass


# Decoder architecture described in Context Encoder paper
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # input size: 256x6x6
        self.fracConv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bNorm1 = nn.BatchNorm2d(128)
        # output size: 128x11x11
        
        # input size: 128x11x11
        self.fracConv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bNorm2 = nn.BatchNorm2d(64)
        # output size: 64X21X21

        # input size: 64x21x21
        self.fracConv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bNorm3 = nn.BatchNorm2d(64)
        # output size: 64x41x41

        # input size: 64x41x41
        self.fracConv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.bNorm4 = nn.BatchNorm2d(32)
        # output size: 32x81x81

        # input size: 32x81x81
        self.fracConv5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1)
        # output size: 3x161x161

        # input size: 3x161x161
        self.resize = nn.Upsample(size=(227, 227), mode='bilinear', align_corners=False)
        # output size: 3x227x227

    def forward(self, x):
        x = self.fracConv1(x)
        x = self.bNorm1(self.relu1(x))

        x = self.fracConv2(x)
        x = self.bNorm2(self.relu2(x))

        x = self.fracConv3(x)
        x = self.bNorm3(self.relu3(x))

        x = self.fracConv4(x)
        x = self.bNorm4(self.relu4(x))

        x = self.fracConv5(x)
        x = self.resize(x)

        return x
    

# Context Encoder model
#   Encoder -> Bottleneck -> Decoder
class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

        self.encoder = Encoder()
        self.bottleNeck = BottleNeck()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleNeck(x)
        x = self.decoder(x)

        return x

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

    def forward(self, x):
        pass