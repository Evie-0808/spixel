# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/16 14:03
@Auth ：
@File ：Superpix.py
@IDE ：PyCharm
@Motto：Talk is cheap. Show me the code

"""
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c


class DilatedConvBlock(chainer.Chain):

    def __init__(self, inout_channel, d_factor, weight=None, bias=None):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D(in_channels=inout_channel, out_channels=inout_channel, ksize=3, stride=1, pad=d_factor,
                                          dilate=d_factor, nobias=False),
            # bn=L.BatchNormalization(64)
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        # h = F.relu(self.bn(self.diconv(x)))
        return h

class Conv(chainer.Chain):
    def __init__(self,batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
        super(Conv, self).__init__(
            conv=L.Convolution2D(in_channels=in_planes, out_channels=out_planes, ksize=kernel_size, stride=stride,
                                 pad=(kernel_size-1)//2, nobias=True),
            bn=L.BatchNormalization(out_planes),

        )
        self.batchNorm = batchNorm
        self.train = True

    def __call__(self, x):
        if self.batchNorm:
            h = F.leaky_relu(self.bn(self.conv(x)), slope=0.1)
        else:
            h = self.bn(self.conv(x))
        # h = F.relu(self.bn(self.diconv(x)))
        return h

class Deconv(chainer.Chain):
    def __init__(self, in_planes, out_planes):
        super(Deconv, self).__init__(
            tconv=L.Deconvolution2D(in_channels=in_planes, out_channels=out_planes, ksize=4, stride=2, pad=1, nobias=False)
        )
    def __call__(self, x):
        return F.leaky_relu(self.tconv(x), slope=0.1)

class Pred_mask(chainer.Chain):
    def __init__(self, inplanes, channel=9):
        super(Pred_mask, self).__init__(
            conv=L.Convolution2D(in_channels=inplanes, out_channels=channel, ksize=3, stride=1, pad=1, nobias=False)
        )
    def __call__(self, x):
        return self.conv(x)


class SuperNet(chainer.Chain, a3c.A3CModel):

    def __init__(self, batchnormal, n_actions):
        w = chainer.initializers.HeNormal()
        # net = CaffeFunction('initial_weight/zhang_cvpr17_denoise_15_gray.caffemodel')
        super(SuperNet, self).__init__(
            conv0a=Conv(batchnormal, 3, 16, 3),
            conv0b=Conv(batchnormal, 16, 16, 3),
            conv1a=Conv(batchnormal, 16, 32, 3, stride=2),
            conv1b=Conv(batchnormal, 32, 32, 3),
            conv2a=Conv(batchnormal, 32, 64, 3, stride=2),
            conv2b=Conv(batchnormal, 64, 64, 3),
            conv3a=Conv(batchnormal, 64, 128, 3, stride=2),
            conv3b=Conv(batchnormal, 128, 128, 3),
            conv4a=Conv(batchnormal, 128, 256, 3, stride=2),
            conv4b=Conv(batchnormal, 256, 256, 3),
            deconv3 = Deconv(256, 128),
            conv3_1 = Conv(batchnormal, 256, 128),
            pred_mask3 = Pred_mask(128, 9),
            deconv2=Deconv(128, 64),
            conv2_1=Conv(batchnormal, 128, 64),
            pred_mask2=Pred_mask(64, 9),
            deconv1=Deconv(64, 32),
            conv1_1=Conv(batchnormal, 64, 32),
            pred_mask1=Pred_mask(32, 9),
            deconv0=Deconv(32, 16),
            conv0_1=Conv(batchnormal, 32, 16),
            pred_mask0=Pred_mask(16, 9),
            diconv5_pi=DilatedConvBlock(9, 3),
            diconv6_pi=DilatedConvBlock(9, 2),
            conv7_pi=chainerrl.policies.SoftmaxPolicy(
                L.Convolution2D(9, n_actions, 3, stride=1, pad=1, nobias=False)),
            diconv5_V=DilatedConvBlock(9, 3),
            diconv6_V=DilatedConvBlock(9, 2),
            conv7_V=L.Convolution2D(9, 1, 3, stride=1, pad=1, nobias=False),
        )
        self.train = True

    def pi_and_v(self, x):
        out1 = self.conv0b(self.conv0a(x))  # 5*5
        out2 = self.conv1b(self.conv1a(out1))  # 11*11
        out3 = self.conv2b(self.conv2a(out2))  # 23*23
        out4 = self.conv3b(self.conv3a(out3))  # 47*47
        out5 = self.conv4b(self.conv4a(out4))  # 95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = F.concat([out4, out_deconv3], 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = F.concat([out3, out_deconv2], 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = F.concat([out2, out_deconv1], 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = F.concat([out1, out_deconv0], 1)
        out_conv0_1 = self.conv0_1(concat0)
        h = self.pred_mask0(out_conv0_1)
        h_pi = self.diconv5_pi(h)
        h_pi = self.diconv6_pi(h_pi)
        pout = self.conv7_pi(h_pi)
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)

        return pout, vout
