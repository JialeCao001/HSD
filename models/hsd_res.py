# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.attention import PAM_Module
from models.model_helper import weights_init


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class FEModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d):
        super(FEModule, self).__init__()
        self.out_channels = out_channels
        inter_channels = in_channels // 4
        self.brancha = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.brancha1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())

        self.sl = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        )
        # self.sl = BasicConv(in_channels, inter_channels+inter_channels, kernel_size=3, padding=1, stride=1)
        self.sn = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )
        self.fuse = nn.Sequential(nn.Dropout2d(0.1, False),
                                     nn.Conv2d(inter_channels + inter_channels + inter_channels, out_channels,
                                               kernel_size=3, stride=stride, padding=1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU())



    def forward(self, x):
        sa_feat = self.sa(self.brancha(x))
        sa_conv = self.brancha1(sa_feat)
        sl_output = self.sl(x)
        sn_output = self.sn(x)
        feat_cat = torch.cat([sa_conv, sl_output, sn_output], dim=1)
        output = self.fuse(feat_cat)
        return output
#
def trans_head():
    arm_trans = []
    arm_trans += [BasicConv(512, 256, kernel_size=3, stride=1, padding=1)]
    arm_trans += [BasicConv(1024, 256, kernel_size=3, stride=1, padding=1)]
    arm_trans += [BasicConv(2048, 256, kernel_size=3, stride=1, padding=1)]
    arm_trans += [BasicConv(2048, 256, kernel_size=3, stride=1, padding=1)]

    orm_trans = []
    orm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]

    return arm_trans, orm_trans

class HSDResnet(nn.Module):
    def __init__(self, block, num_blocks, size):
        super(HSDResnet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.extras0 = self._make_layer(block, 512, 2, stride=2)
        
        self.fe1 = FEModule(512,256)
        self.fe2 = FEModule(512,256)
        self.fe3 = FEModule(512,256)
        self.arm_trans = nn.ModuleList(trans_head()[0])
        self.orm_trans = nn.ModuleList(trans_head()[1])

        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras0.apply(weights_init)
        self.arm_trans.apply(weights_init)
        self.orm_trans.apply(weights_init)
        self.fe1.apply(weights_init)
        self.fe2.apply(weights_init)
        self.fe3.apply(weights_init)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.extras0(c5)

        c3_0 = self.arm_trans[0](c3)
        c4_0 = self.arm_trans[1](c4)
        c5_0 = self.arm_trans[2](c5)
        c6_0 = self.arm_trans[3](c6)

        arm_sources = [c3_0, c4_0, c5_0, c6_0]

        odm_sources = []
        up = F.upsample(arm_sources[1], size=arm_sources[0].size()[2:], mode='bilinear')
        odm_sources.append(self.fe1(torch.cat([up, arm_sources[0]], dim = 1)))
        up = F.upsample(arm_sources[2], size=arm_sources[1].size()[2:], mode='bilinear')
        odm_sources.append(self.fe2(torch.cat([up, arm_sources[1]], dim=1)))
        up = F.upsample(arm_sources[3], size=arm_sources[2].size()[2:], mode='bilinear')
        odm_sources.append(self.fe3(torch.cat([up, arm_sources[2]], dim=1)))
        odm_sources.append(self.orm_trans[3](arm_sources[3]))

        return arm_sources, odm_sources


def HSDResnet50(size):
    return HSDResnet(Bottleneck, [3, 4, 6, 3], size)


def HSDResnet101(size):
    return HSDResnet(Bottleneck, [3, 4, 23, 3], size)


def HSDResnet152(size):
    return HSDResnet(Bottleneck, [3, 8, 36, 3], size)
