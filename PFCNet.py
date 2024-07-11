import torch
import numpy as np
from torch.nn import functional as F
import random
from backbone.Shunted.SSA import *
from model import *
import torch.nn as nn
import torch.fft as fft
from math import sqrt
import math

channel_lay1 = 64
channel_lay2 = 128
channel_lay3 = 256
channel_lay4 = 512

def DOWN1(in_, out_):
    return nn.Sequential(
        nn.Conv2d(in_, out_, 3, 1, 1),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    )

class HL(nn.Module):
    def __init__(self, inch1,inch2,h1,h2):
        super(HL, self).__init__()
        self.msa1 = MSAL(inch1, h1, h1)
        self.msa2 = MSAL(inch2, h2, h2)
        self.cb1 = convblock(inch1, inch1, 3, 1, 1)
        self.cb2 = convblock(inch2, inch2, 3, 1, 1)
        self.up = DOWN1(inch1,inch2)
        self.cbr1 = nn.Sequential(
            convblock(inch1, inch2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.cbr2 = convblock(inch2, inch2, 3, 1, 1)
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        self.conv = nn.Conv2d(inch2,inch2,1,1,0)
        self.sig = nn.Sigmoid()

    def forward(self, H,L):
        H1 = self.a * self.sig(self.msa1(H)) * H + H
        H2 = self.cb1(H1)
        H3 = self.up(H2)+ self.cbr1(H)
        L1 = self.b * self.sig(self.msa2(L)) * L + L
        L2 = self.cb2(L1)
        x = self.conv(H3 + L2)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.HL1 = HL(channel_lay4, channel_lay3, 10, 20)
        self.HL2 = HL(channel_lay3, channel_lay2, 20, 40)
        self.HL3 = HL(channel_lay2, channel_lay1, 40, 80)

    def forward(self,f4,f3,f2,f1):
        x1 = self.HL1(f4, f3)
        x2 = self.HL2(x1, f2)
        x3 = self.HL3(x2, f1)
        return x3,x2,x1

#########################################

class FAG(nn.Module):
    def __init__(self, in_channels, cutoff_ratio=0.1):
        super(FAG, self).__init__()
        self.cutoff_ratio = cutoff_ratio
        self.relu = nn.ReLU()
        # self.cb1 = convblock(512,512,3,1,1)

    def fenli(self,x):
        image_fft = fft.fftn(x, dim=(-2, -1))                                                            # 对输入图像进行二维离散傅里叶变换
        height, width = x.shape[-2:]                                                                     # 取图像的高度和宽度。计算图像的中心位置
        center_height = height // 2
        center_width = width // 2                                                                        # 计算图像的中心位置
        low_frequency = image_fft.clone()                                                                # 创建了 image_fft 的一个副本，用于存储低频部分。
        low_frequency[:, :, center_height - 5:center_height + 5, center_width - 5:center_width + 5] = 0  # 将图像的中心区域（即低频部分）之外的频率分量设置为0。这里，中心区域的大小为10x10，即中心点周围5个像素宽的区域。
        high_frequency = image_fft - low_frequency                                                       # 高频部分为原始频谱减去低频部分

        return low_frequency, high_frequency

    def forward(self, x):
        # x = self.relu(x) + 1e-8
        xlow_frequency, xhigh_frequency = self.fenli(x)

        # 计算每个通道层次的特征统计量的方差，并建模为高斯分布
        low_frequency_stats = torch.var(xlow_frequency, dim=(0, 2, 3), keepdim=True)  # 计算特征统计量的方差
        gaussian_distribution = torch.distributions.Normal(0, torch.sqrt(low_frequency_stats+ 1e-8))  # 使用方差建模为高斯分布

        # 重采样得到扰动后的统计量，并使用扰动后的统计量来重构低频谱
        sampled_low_freq_stats = gaussian_distribution.sample()                             # 从高斯分布中采样扰动后的统计量
        reconstructed_low_frequency = sampled_low_freq_stats.expand(xlow_frequency.size())  # 使用扰动后的统计量来重构低频谱

        # 将重构的低频谱与原始的高频谱结合
        augmented_spectrum = reconstructed_low_frequency + xhigh_frequency
        augmented_spectrum = augmented_spectrum

        return augmented_spectrum


class CC1(nn.Module):
    def __init__(self, inch1,inch2):
        super(CC1, self).__init__()
        self.convr1 = nn.Conv2d(inch1,inch1,1,1,0)
        self.convd1 = nn.Conv2d(inch1, inch1, 1, 1, 0)
        self.car1 = RGA_Module_S(inch1,inch2)
        self.cad1 = RGA_Module_S(inch1,inch2)
        self.fa = FA(inch1)
        self.cb1 = convblock(inch1,inch1,3,1,1)
        self.convx = nn.Conv2d(inch1,inch1,3,1,1)
        self.sig = nn.Sigmoid()
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, R,D):
        r1 = self.convr1(R)
        d1 = self.convd1(D)
        r2 = self.car1(r1) * r1
        d2 = self.cad1(d1) * d1
        d2 = r2 * d2 + d1 + r1
        x1 = self.a * (self.sig(self.convx(self.fa(d2).float()))) * d2 + d2
        x2 = self.cb1(x1)

        return x2

class CC2(nn.Module):
    def __init__(self, inch1,inch2):
        super(CC2, self).__init__()
        self.convr1 = nn.Conv2d(inch1,inch1,1,1,0)
        self.convd1 = nn.Conv2d(inch1, inch1, 1, 1, 0)
        self.car1 = RGA_Module_C(inch1,inch2)
        self.cad1 = RGA_Module_C(inch1,inch2)
        self.fa = FAG(inch1)
        self.cb1 = convblock(inch1,inch1,3,1,1)
        self.convx = nn.Conv2d(inch1,inch1,3,1,1)
        self.sig = nn.Sigmoid()
        self.a = nn.Parameter(torch.ones(1))


    def forward(self, R,D):
        r1 = self.convr1(R)
        d1 = self.convd1(D)
        r2 = self.car1(r1) * r1
        d2 = self.cad1(d1) * d1
        d2 = r2 * d2 + d1 + r1
        x1 = self.a * (self.sig(self.convx(self.fa(d2).float()))) * d2 + d2
        x2 = self.cb1(x1)

        return x2

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class Bup(nn.Module):
    def __init__(self,inch1,inch2 ):
        super(Bup, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inch1, inch2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, y):
        y1 = self.conv1(y)
        return y1

class shuntbase(nn.Module):
    def __init__(self):
        super(shuntbase, self).__init__()
        self.rgb_net = shunted_b(pretrained=True)
        self.d_net = shunted_b(pretrained=True)

        self.cc4 = CC1(channel_lay4, 100)
        self.cc3 = CC1(channel_lay3, 400)
        self.cc2 = CC2(channel_lay2, 1600)
        self.cc1 = CC2(channel_lay1, 6400)
        self.decoder = Decoder()
        ########                   jd              #######################
        self.jds = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.jdy2 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.jdy3 = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        )
        self.jdx1 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.jdx2 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.jdx3 = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        )
        self.jdx4 = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True),
        )

    def forward(self, rgb, d):
        d = torch.cat((d, d, d), dim=1)
        rgb_list = self.rgb_net(rgb)
        depth_list = self.d_net(d)

        r1 = rgb_list[0]
        r2 = rgb_list[1]
        r3 = rgb_list[2]
        r4 = rgb_list[3]

        d1 = depth_list[0]
        d2 = depth_list[1]
        d3 = depth_list[2]
        d4 = depth_list[3]

        fus4 = self.cc4(r4,d4)
        fus3 = self.cc3(r3,d3)
        fus2 = self.cc2(r2,d2)
        fus1 = self.cc1(r1,d1)

        y1,y2,y3 = self.decoder(fus4,fus3,fus2,fus1)
        outs = self.jds(y1)
        outy2 = self.jdy2(y2)
        outy3 = self.jdy3(y3)

        out1 = self.jdx1(fus1)
        out2 = self.jdx2(fus2)
        out3 = self.jdx3(fus3)
        out4 = self.jdx4(fus4)

        return outs,out1,out2,out3,out4,outy2,outy3

    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.rgb_net.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.rgb_net.load_state_dict(model_dict_r)

        model_dict_d = self.d_net.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_d.update(state_dict_d)
        self.d_net.load_state_dict(model_dict_d)
        print('self.rgb_uniforr loading', 'self.depth_unifor loading')

if __name__ == '__main__':
    rgb = torch.randn([1, 3, 320, 320]).cuda()                                   # batch_size=1，通道3，图片尺寸320*320
    depth = torch.randn([1, 1, 320, 320]).cuda()
    model = shuntbase().cuda()
    a = model(rgb, depth)
    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)
    print(a[3].shape)
    print(a[4].shape)
    print(a[5].shape)
    print(a[6].shape)
