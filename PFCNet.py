import torch
import numpy as np
from torch.nn import functional as F
import random
from backbone.Shunted.SSA import *
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

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MSAL(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=2, freq_sel_method='top16'):
        super(MSAL, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 4) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 4) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        z = x * y.expand_as(x)

        return z

class MultiSpectralDCTLayer(nn.Module):

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)

        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter

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

class FA(nn.Module):
    def __init__(self,mask_radio=0.1,cutoff_ratio=0.1,noise_mode=1, perturb_prob=0.5,
                 uncertainty_factor=1.0,noise_layer_flag=1,):
        super(FA, self).__init__()
        self.cutoff_ratio = cutoff_ratio
        self.relu = nn.ReLU()
        self.mask_radio = mask_radio
        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag
        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.p = perturb_prob
    def _reparameterize(self, mu, std, epsilon_norm):
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spectrum_noise(self, img_fft, ratio=1.0,):
        if random.random() > self.p:
            return img_fft
        batch_size, h, w, c = img_fft.shape

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        img_abs = torch.fft.fftshift(img_abs, dim=(1))

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = 0
        img_abs_ = img_abs.clone()

        miu = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),keepdim=True)
        var = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),keepdim=True)
        sig = (var + self.eps).sqrt()  # Bx1x1xC
        var_of_miu = torch.var(miu, dim=0, keepdim=True)
        var_of_sig = torch.var(sig, dim=0, keepdim=True)
        sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
        sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
        epsilon_norm_sig = torch.randn_like(sig_of_sig)

        miu_mean = miu
        sig_mean = sig

        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)

        img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = gamma * (img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] - miu) / sig + beta

        img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = x.to(torch.float32)
        x = fft.fftn(x, dim=(-2, -3), norm='ortho')
        x = self.spectrum_noise(x, ratio=self.mask_radio)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.reshape(B, C, H, W)

        return x

class RGA_Module_C(nn.Module):
    def __init__(self, in_channel,in_spatial, cha_ratio=4, spa_ratio=4, down_ratio=2):
        super(RGA_Module_C, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial
        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        self.gx_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )

        self.gg_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        num_channel_c = 1 + self.inter_channel
        self.W_channel = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_c // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        self.theta_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )
        self.phi_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)             #1,100，512，1
        xc1 = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
        xc2 = self.phi_channel(xc).squeeze(-1)
        Gc = torch.matmul(xc1, xc2)
        Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
        Gc_out = Gc.unsqueeze(-1)
        Gc_joint = torch.cat((Gc_in, Gc_out), 1)
        Gc_joint = self.gg_channel(Gc_joint)

        g_xc = self.gx_channel(xc)
        g_xc = torch.mean(g_xc, dim=1, keepdim=True)
        yc = torch.cat((g_xc, Gc_joint), 1)
        W_yc = self.W_channel(yc).transpose(1, 2)
        out = F.sigmoid(W_yc) * x + x

        return out

class RGA_Module_S(nn.Module):
    def __init__(self, in_channel, in_spatial, use_channel=True,cha_ratio=4, spa_ratio=4, down_ratio=16):
        super(RGA_Module_S, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial
        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        self.gx_spatial = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
            )
        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_spatial * 2, in_spatial,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_spatial),
            nn.ReLU()
        )
        num_channel_s = 1 + in_spatial
        self.W_spatial = nn.Sequential(
            nn.Conv2d(num_channel_s, in_spatial,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_spatial),
            nn.ReLU(),
            nn.Conv2d(in_spatial, 1,kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
            )
        self.theta_spatial = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.phi_spatial = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        theta_xs = self.theta_spatial(x)
        phi_xs = self.phi_spatial(x)
        theta_xs = theta_xs.view(b, self.in_channel, -1)
        theta_xs = theta_xs.permute(0, 2, 1)
        phi_xs = phi_xs.view(b, self.in_channel, -1)
        Gs = torch.matmul(theta_xs, phi_xs)                      #B,HW,HW
        Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
        Gs_out = Gs.view(b, h * w, h, w)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)
        Gs_joint = self.gg_spatial(Gs_joint)               #B,HW,H,W
        g_xs = self.gx_spatial(x)
        g_xs = torch.mean(g_xs, dim=1, keepdim=True)
        ys = torch.cat((g_xs, Gs_joint), 1)

        W_ys = self.W_spatial(ys)
        x = F.sigmoid(W_ys.expand_as(x)) * x + x
        return x

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