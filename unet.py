"""U-net generator

This module provides a UNet class to be used as the generator model in the algorithm
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

ACTIVATION = nn.ReLU

class Identity(nn.Module):
    def forward(self, x):
        return x

def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

def conv2d_bn_block(in_channels, out_channels, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block conv-bn-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor = self.scale_factor, mode='bilinear', align_corners=True)
    
def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=0, momentum=0.01, activation=ACTIVATION, dimensions = 2):
    '''
    returns a block deconv-bn-activation
    use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        if dimensions == 2:
            conv_layer = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        elif dimensions == 3:
            conv_layer = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        up = nn.Sequential(
            Upsample(scale_factor=2),
            conv_layer
        )
    else:
        if dimensions == 2:
            up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
        elif dimensions == 3:
            up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    if dimensions == 2:
        bn_layer = nn.BatchNorm2d(out_channels, momentum=momentum)
    elif dimensions == 3:
        bn_layer = nn.BatchNorm3d(out_channels, momentum=momentum)
    return nn.Sequential(
        up,
        bn_layer,
        activation(),
    )

def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block linear-bn-activation
    '''
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )

def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block conv-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, nf=16, batch_norm=True, dimensions=2, use_extra_inputs = True):
        super(UNet, self).__init__()
        self.dimensions = dimensions
        conv_block = conv2d_bn_block if batch_norm else conv2d_block
        max_pool = nn.MaxPool2d(2) if int(dimensions) is 2 else nn.MaxPool3d(2)
        act = torch.nn.ReLU
        self.down0 = nn.Sequential(
            conv_block(n_channels, nf, activation=act),
            conv_block(nf, nf, activation=act)
        )
        self.down1 = nn.Sequential(
            max_pool,
            conv_block(nf, 2*nf, activation=act),
            conv_block(2*nf, 2*nf, activation=act),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2*nf, 4*nf, activation=act),
            conv_block(4*nf, 4*nf, activation=act),
        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4*nf, 8*nf, activation=act),
            conv_block(8*nf, 8*nf, activation=act),
        )
        self.down4 = nn.Sequential(
            max_pool,
            conv_block(8*nf, 16*nf, activation=act),
            conv_block(16*nf, 16*nf, activation=act),
        )

        self.up4 = deconv2d_bn_block(16*nf+3, 8*nf, activation=act)

        self.conv4 = nn.Sequential(
            conv_block(16*nf, 8*nf, activation=act),
            conv_block(8*nf, 8*nf, activation=act),
        )
        self.up3 = deconv2d_bn_block(8*nf, 4*nf, activation=act)
        
        self.conv5 = nn.Sequential(
            conv_block(8*nf, 4*nf, activation=act),
            conv_block(4*nf, 4*nf, activation=act),
        )
        self.up2 = deconv2d_bn_block(4*nf, 2*nf, activation=act, dimensions = dimensions)

        self.conv6 = nn.Sequential(
            conv_block(4*nf, 2*nf, activation=act),
            conv_block(2*nf, 2*nf, activation=act),
        )
        self.up1 = deconv2d_bn_block(2*nf, nf, activation=act, dimensions = dimensions)

        self.conv7 = nn.Sequential(
            conv_block(2*nf, nf, activation=act),
            conv_block(nf, 1, activation=Identity),
        )
        
    def forward(self, x, desired_output, groundtruth_regression):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        desired_output = desired_output.view(-1,1,1,1).expand([x4.size(0),1,x4.size(2),x4.size(3)])
        groundtruth_regression = groundtruth_regression.view(-1,1,1,1).expand([x4.size(0),1,x4.size(2),x4.size(3)])
        x5 = torch.cat([x4, desired_output, groundtruth_regression, (groundtruth_regression - desired_output)], dim = 1)
        
        xu4 = self.up4(x5)
        cat3 = crop_and_concat(xu4, x3)
        x10 = self.conv4(cat3)

        xu3 = self.up3(x10)
        cat3 = crop_and_concat(xu3, x2)
        x11 = self.conv5(cat3)
        xu2 = self.up2(x11)
        cat2 = crop_and_concat(xu2, x1)
        x12 = self.conv6(cat2)
        xu1 = self.up1(x12)
        cat1 = crop_and_concat(xu1, x0)
        x13 = self.conv7(cat1)
        x14 = x13
        return x14
