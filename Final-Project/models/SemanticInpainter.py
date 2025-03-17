# Author: Brett Kinsella and Rohan Gujral
# Date: 3/9/2025
# Description:
# Implementation of the context encoder semantic inpainting model trained using joint reconstruction loss (L2) and adversarial loss from the paper "Context Encoders: Feature Learning by Inpainting"
#   by Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros
#   https://arxiv.org/abs/1604.07379

import torch.nn as nn

# Encoder
# Listed dimensions are  C X H x W
# Designed for 128x128 input images
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # input size: 3x128x128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # output size: 64x64x64

        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm2 = nn.BatchNorm2d(64)
        # output size: 64x32x32

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm3 = nn.BatchNorm2d(128)
        # output size: 128x16x16

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm4 = nn.BatchNorm2d(256)
        # output size: 256x8x8

        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm5 = nn.BatchNorm2d(512)
        # output size: 512x4x4

        self.conv6 = nn.Conv2d(512, 4000, kernel_size=4)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm6 = nn.BatchNorm2d(4000)
        # output size: 4000x1x1

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

        x = self.conv6(x)
        x = self.bNorm6(self.relu6(x))

        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # input size: 4000x1x1
        self.fracConv1 = nn.ConvTranspose2d(4000, 512, kernel_size=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bNorm1 = nn.BatchNorm2d(512)
        # output size: 512x4x4
        
        self.fracConv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bNorm2 = nn.BatchNorm2d(256)
        # output size: 256x8x8
        
        self.fracConv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bNorm3 = nn.BatchNorm2d(128)
        # output size: 128x16x16

        self.fracConv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.bNorm4 = nn.BatchNorm2d(64)
        # output size: 64x32x32

        self.fracConv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh5 = nn.Tanh()
        # output size: 3x64x64

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
        x = self.tanh5(x)

        return x
    

# Context Encoder model
#   Encoder -> Bottleneck -> Decoder
class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

# Adversarial discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # input size: 3x64x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # output size: 64x32x32

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm2 = nn.BatchNorm2d(128)
        # output size: 128x16x16

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm3 = nn.BatchNorm2d(256)
        # output size: 256x8x8

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm4 = nn.BatchNorm2d(512)
        # output size: 512x4x4

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4)
        self.sigmoid = nn.Sigmoid()
        # output size: 1x1x1

        self.flatten = nn.Flatten()
        # output size: 1

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
        x = self.sigmoid(x)

        x = self.flatten(x)

        return x