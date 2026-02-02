from .DWT_IDWT_layer import *
from .squeeze_and_excitation_3D import ProjectExciteLayer
from .attention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [(np.array(krnlsz) * 0 + 1) * half_dim] + [krnlsz] * 2
    else:
        outsz = [krnlsz]
    return tuple(outsz)


class StackConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mid_channels=None,
                 model_type='3d', residualskip=False, device=None, dtype=None):
        super(StackConvBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.residualskip = residualskip
        padding = {'same': kernel_size // 2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channels = out_channels if mid_channels is None else mid_channels

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)

        self.short_cut_conv = self.ConvBlock(in_channels, out_channels, 1, extdim(stride))
        self.norm0 = self.InstanceNorm(out_channels, affine=True)
        self.conv1 = self.ConvBlock(in_channels, mid_channels, extdim(kernel_size, 3), extdim(stride),
                                    padding=extdim(padding, 1), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(mid_channels, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = self.ConvBlock(mid_channels, out_channels, extdim(kernel_size, 3), extdim(1),
                                    padding=extdim(padding, 1), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(out_channels, affine=True, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.relu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out_res = self.relu2(o_c2 + short_cut_conv)
        else:
            out_res = self.relu2(o_c2)
        return out_res


class WATConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dwt_channels=1, kernel_size=3, stride=2, model_type='3d',
                 residualskip=True):
        super(WATConvBlock3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = DWT_3D("haar")
        self.resblock = StackConvBlock(in_channels, out_channels, kernel_size, stride, model_type=model_type,
                                       residualskip=residualskip)
        self.peblock = ProjectExciteLayer(out_channels + 7 * dwt_channels)
        self.conv = nn.Conv3d(out_channels + 7 * dwt_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        # 进行小波变换
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x2)
        # 残差结构处理
        o1 = self.resblock(x1)
        wat_tensor = torch.cat([o1, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        o2 = self.peblock(wat_tensor)
        o3 = self.conv(o2) + o1
        return o3, LLL


class WATConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dwt_channels=1, kernel_size=3, stride=2, model_type='2d',
                 residualskip=True):
        super(WATConvBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = DWT_2D("haar")
        self.resblock = StackConvBlock(in_channels, out_channels, kernel_size, stride, model_type=model_type,
                                       residualskip=residualskip)
        self.ca = CoordAtt(out_channels + 3 * dwt_channels, out_channels + 3 * dwt_channels)
        self.conv = nn.Conv2d(out_channels + 3 * dwt_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        # 进行小波变换
        LL, LH, HL, HH = self.dwt(x2)
        # 残差结构处理
        o1 = self.resblock(x1)
        wat_tensor = torch.cat([o1, LH, HL, HH], dim=1)
        o2 = self.ca(wat_tensor)
        o3 = self.conv(o2) + o1
        return o3, LL


class WatEncoder(nn.Module):
    def __init__(self, in_channels=1, basedim=16, downdeepth=2, model_type='3d'):
        super().__init__()
        if model_type == '3d':
            self.conv = nn.Conv3d
        else:
            self.conv = nn.Conv2d

        self.begin_conv = nn.Sequential(
            self.conv(in_channels, basedim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(basedim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        if model_type == '3d':
            self.encoding_block = nn.ModuleList([
                WATConvBlock3D(basedim * 2 ** convidx,
                               basedim * 2 ** (convidx + 1), dwt_channels=in_channels,
                               kernel_size=3, stride=2, model_type=model_type)
                for convidx in range(0, downdeepth)
            ])
        else:
            self.encoding_block = nn.ModuleList([
                WATConvBlock2D(basedim * 2 ** convidx,
                               basedim * 2 ** (convidx + 1), dwt_channels=in_channels,
                               kernel_size=3, stride=2, model_type=model_type)
                for convidx in range(0, downdeepth)
            ])
        self.enc_out = basedim * 2 ** downdeepth
        # self.pre_vq_conv = self.conv(self.enc_out, embedding_dim, 3, 1, padding=1)

    def forward(self, x):
        o1 = self.begin_conv(x)
        L = x
        # features = []
        for block in self.encoding_block:
            o1, L = block(o1, L)
            # features.append(o1)
        # o2 = self.pre_vq_conv(o1)
        return o1


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dwt_channels=1, kernel_size=3, stride=2, model_type='2d',
                 residualskip=True):
        super(ConvBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resblock = StackConvBlock(in_channels, out_channels, kernel_size, stride, model_type=model_type,
                                       residualskip=residualskip)
        self.ca = CoordAtt(out_channels, out_channels)

    def forward(self, x):
        # 残差结构处理
        o1 = self.resblock(x)
        o2 = self.ca(o1)
        return o2


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dwt_channels=1, kernel_size=3, stride=2, model_type='3d',
                 residualskip=True):
        super(ConvBlock3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resblock = StackConvBlock(in_channels, out_channels, kernel_size, stride, model_type=model_type,
                                       residualskip=residualskip)
        self.peblock = ProjectExciteLayer(out_channels)

    def forward(self, x):
        # 残差结构处理
        o1 = self.resblock(x)
        o2 = self.peblock(o1)
        return o2


class ConEncoder(nn.Module):
    def __init__(self, in_channels=1, basedim=16, downdeepth=2, model_type='2d', embedding_dim=8):
        super().__init__()
        if model_type == '3d':
            self.conv = nn.Conv3d
        else:
            self.conv = nn.Conv2d

        self.begin_conv = nn.Sequential(
            self.conv(in_channels, basedim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(basedim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        if model_type == '3d':
            self.encoding_block = nn.ModuleList([
                ConvBlock3D(basedim * 2 ** convidx,
                            basedim * 2 ** (convidx + 1), dwt_channels=in_channels,
                            kernel_size=3, stride=2, model_type=model_type)
                for convidx in range(0, downdeepth)
            ])
        else:
            self.encoding_block = nn.ModuleList([
                ConvBlock2D(basedim * 2 ** convidx,
                            basedim * 2 ** (convidx + 1), dwt_channels=in_channels,
                            kernel_size=3, stride=2, model_type=model_type)
                for convidx in range(0, downdeepth)
            ])
        self.enc_out = basedim * 2 ** downdeepth
        # self.pre_vq_conv = self.conv(self.enc_out, embedding_dim, 3, 1, padding=1)

    def forward(self, x):
        o1 = self.begin_conv(x)
        # features = []
        for block in self.encoding_block:
            o1 = block(o1)
            # features.append(o1)
        # o2 = self.pre_vq_conv(o1)
        return o1


if __name__ == "__main__":
    # model = WatEncoder(in_channels=1, basedim=16, downdeepth=2, model_type='3d', embedding_dim=8)
    # x = torch.randn(1, 1, 64, 64, 64)
    # y = model(x)
    # print(y.shape)

    model = WatEncoder(in_channels=1, basedim=16, downdeepth=4, model_type='2d', embedding_dim=8).cuda()
    x = torch.randn(1, 1, 224, 224).cuda()
    y, features = model(x)
    print(y.shape)
    for f in features:
        print(f.shape)