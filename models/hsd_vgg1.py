# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from models.model_helper import weights_init
from models.attention import PAM_Module


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            x) * x
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=2, dilation=2),
            BasicConv(inter_planes, out_planes, kernel_size=(3, 3), stride=1, padding=(1, 1))
        )

    def forward(self, x):
        out = self.single_branch(x)
        return out
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

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



def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5, conv6,
        nn.ReLU(inplace=True), conv7,
        nn.ReLU(inplace=True)
    ]
    return layers


base = {
    '300': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
    '512': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
}


def add_extras(size):
    layers = []
    layers += [BasicBlock(1024, 256, stride=2)]
    layers += [BasicBlock(256, 256, stride=2)]
    return layers

#

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * (1+y)

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

        # aspp
        self.sl = BasicConv(in_channels, inter_channels+inter_channels, kernel_size=3, padding=1, stride=1)
        self.sn = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )

        self.fuse = nn.Sequential(nn.Dropout2d(0.1, False),
                                     nn.Conv2d(inter_channels + inter_channels + inter_channels+inter_channels, out_channels,
                                               kernel_size=3, stride=stride, padding=1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU())



    def forward(self, x):
        sa_feat = self.sa(self.brancha(x))
        sa_conv = self.brancha1(sa_feat)

        sl_output = self.sl(x)
        sn_output = self.sn(x)

        feat_cat = torch.cat([sa_conv, sl_output, sn_output], dim=1)
        sasc_output = self.fuse(feat_cat)

        return sasc_output

def trans_head():
    arm_trans = []
    arm_trans += [BasicConv(512, 256, kernel_size=3, stride=1, padding=1)]
    arm_trans += [BasicConv(1024, 256, kernel_size=3, stride=1, padding=1)]
    arm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]
    arm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]

    orm_trans = []
    orm_trans += [BasicConv(256, 512, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 512, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 512, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]

    return arm_trans, orm_trans

class VGG16Extractor(nn.Module):
    def __init__(self, size, channel_size='48'):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(base[str(size)], 3))
        self.extras = nn.ModuleList(add_extras(str(size)))

        self.fe1 = FEModule(512,256)
        self.fe2 = FEModule(512,256)
        self.fe3 = FEModule(512,256)
        self.arm_trans = nn.ModuleList(trans_head()[0])
        self.orm_trans = nn.ModuleList(trans_head()[1])

        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.orm_trans.apply(weights_init)
        self.arm_trans.apply(weights_init)
        self.fe1.apply(weights_init)
        self.fe2.apply(weights_init)
        self.fe3.apply(weights_init)


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_sources = list()

        for i in range(23):
            x = self.vgg[i](x)
        #38x38
        c2 = x
        c2 = self.arm_trans[0](c2)
        arm_sources.append(c2)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        #19x19
        c3 = x
        c3 = self.arm_trans[1](c3)
        arm_sources.append(c3)

        # 10x10
        x = self.extras[0](x)

        # c4 = x
        c4 = self.arm_trans[2](x)
        arm_sources.append(c4)

        # 5x5
        x = self.extras[1](x)
        # c5 = x
        c5 = self.arm_trans[3](x)
        arm_sources.append(c5)

        odm_sources = []
        up = F.upsample(arm_sources[1], size=arm_sources[0].size()[2:], mode='bilinear')
        odm_sources.append(self.fe1(torch.cat([up, arm_sources[0]], dim = 1)))
        up = F.upsample(arm_sources[2], size=arm_sources[1].size()[2:], mode='bilinear')
        odm_sources.append(self.fe2(torch.cat([up, arm_sources[1]], dim=1)))
        up = F.upsample(arm_sources[3], size=arm_sources[2].size()[2:], mode='bilinear')
        odm_sources.append(self.fe3(torch.cat([up, arm_sources[2]], dim=1)))
        odm_sources.append(self.orm_trans[3](arm_sources[3]))


        return arm_sources, odm_sources


def hsd_vgg(size):
    return VGG16Extractor(size)