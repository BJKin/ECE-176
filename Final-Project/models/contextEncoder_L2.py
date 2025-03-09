# Author: Brett Kinsella
# Date: 3/9/2025
# Description:
# Implementation of the Context Encoder model trained using reconstruction loss (L2) from the paper "Context Encoders: Feature Learning by Inpainting"
#   by Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros
#   https://arxiv.org/abs/1604.07379

import torch.nn as nn


# ALEXNET architecture used for the encoder with added batch normalization layers
# Leakky ReLu activation functions are used instead of ReLu with a negative slope of 0.2
# Listed dimensions are  C X H x W
# Designed for 227x227 input images
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # input size: 3x227x227
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.bNorm1 = nn.BatchNorm2d(96)
        # output size: 96x27x27

        # input size: 96x27x27
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.bNorm2 = nn.BatchNorm2d(256)
        # output size: 256x13x13

        # input size: 256x13x13
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm3 = nn.BatchNorm2d(384)
        # output size: 384x13x13

        # input size: 384x13x13
        self.conv4 = nn.Conv2d(384, 192, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bNorm4 = nn.BatchNorm2d(192)
        # output size: 192x13x13

        # input size: 192x13x13
        self.conv5 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.pool5 = nn.MaxPool2d(3, 2)
        self.bNorm5 = nn.BatchNorm2d(256)
        # output size: 256x6x6

    def forward(self, x):
        x = self.conv1(x)
        x = self.bNorm1(self.pool1(self.relu1(x)))

        x = self.conv2(x)
        x = self.bNorm2(self.pool2(self.relu2(x)))

        x = self.conv3(x)
        x = self.bNorm3(self.relu3(x))

        x = self.conv4(x)
        x = self.bNorm4(self.relu4(x))

        x = self.conv5(x)
        x = self.bNorm5(self.pool5(self.relu5(x)))

        return x
    

# Channel-wise fully connected layer followed by a 1x1 convolution layer
# Implementaion of the channelwise fully connected layer was unclear in the original paper, so 
#   a 1x1 depth-wise convolutional layer w` `   `s used to create the same effect
class ChannelWiseFC(nn.Module):
    def __init__(self):
        super(ChannelWiseFC, self).__init__()
        # input size: 256x6x6
        self.channel_wise_fc = nn.Conv2d(256, 256, kernel_size=1, groups=256, bias=True)
        self.cross_channel_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        # output size: 256x6x6

        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.channel_wise_fc(x)
        x = self.cross_channel_conv(x)
        x = self.dropout(x)

        return x


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
#   Encoder -> ChannelWiseFC -> Decoder
class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

        self.encoder = Encoder()
        self.channelWiseFC = ChannelWiseFC()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.channelWiseFC(x)
        x = self.decoder(x)

        return x

