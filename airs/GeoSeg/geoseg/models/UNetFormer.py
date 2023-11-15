import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm

import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, Conv3d
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):#nn.BatchNorm2d对输入的四维数组进行批量标准化处理
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential): #卷积层加norm    卷积层之后添加BatchNorm2d进行数据的归一化处理
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential): # 可分离卷积 降低计算复杂度
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):#多层感知机模块 优化网络
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)#卷积层
        self.act = act_layer()#  act 激活函数
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)#Dropout主要的作用是在神经网络训练过程中防止模型过拟合

    def forward(self, x):
        x = self.fc1(x)
        # torch.Size([16, 256, 32, 32])

        x = self.act(x)
        # torch.Size([16, 256, 32, 32])

        x = self.drop(x)
        # torch.Size([16, 256, 32, 32])
       
        x = self.fc2(x)
        # torch.Size([16, 64, 32, 32])
        x = self.drop(x)
        # torch.Size([16, 64, 32, 32])

        return x


class GlobalLocalAttention(nn.Module):#全局注意力
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
	
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        # print('1111111111111111111')
        # torch.Size([16, 64, 32, 32])
       
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        # torch.Size([16, 64, 32, 32])
        # print(x.shape)
        # print('1111111111111111111')
        # exit()
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)
 
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale #矩阵相乘

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v # softmax输出 与 矩阵v相乘  

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        # torch.Size([16, 64, 32, 32])
        # print(out.shape)
        # print('1111111111111111111')
        # exit()

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]
        # torch.Size([16, 64, 32, 32])
        # print(out.shape)
        # print('1111111111111111111')
        # exit()

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        # torch.Size([16, 64, 32, 32])
        # print(x.shape)
        # print('1111111111111111111111')
       
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # torch.Size([16, 64, 32, 32])
        # print(x.shape)
        # print('1111111111111111111111')
        # exit()
        
       
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):#特征细化头
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #interroplate 利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        # torch.Size([16, 64, 256, 256])
        x = self.post_conv(x)
        # torch.Size([16, 64, 256, 256])
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        # torch.Size([16, 64, 256, 256])
        # print(pa.shape)
        # print('1111111111111111111')
        ca = self.ca(x) * x
        # torch.Size([16, 64, 256, 256])
        # print(pa.shape)
        # print('1111111111111111111')
        # exit()
        x = pa + ca
        # torch.Size([16, 64, 256, 256])
        x = self.proj(x) + shortcut
        # torch.Size([16, 64, 256, 256])
        x = self.act(x)
        # torch.Size([16, 64, 256, 256])

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            # up
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)
            # torch.Size([16, 64, 64, 64])
            # print(h4.shape)
            # print('1111111111111111111111111111111')
            # exit()

            # torch.Size([16, 64, 16, 16])
            # print(x.shape)
            # exit()
            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)
            # h3  torch.Size([16, 64, 64, 64])
            # torch.Size([16, 64, 32, 32])
            # print(x.shape)
            # print('1111111111111111111111111111111')
            # exit()
            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            # torch.Size([16, 64, 64, 64])
            # print(x.shape)
            # print('1111111111111111111111111111111')
            # exit()
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            # torch.Size([16, 6, 512, 512])

            ah = h4 + h3 + h2
            # torch.Size([16, 64, 64, 64])
            # print(ah.shape)
            # print('1111111111111111111111111111111')
            # exit()
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            # torch.Size([16, 64, 32, 32])
            x = self.p3(x, res3)
            # torch.Size([16, 64, 64, 64])

            x = self.b3(x)
            # torch.Size([16, 64, 64, 64])
           

            x = self.p2(x, res2)
            # torch.Size([16, 64, 128, 128])

           
            x = self.b2(x)
            # torch.Size([16, 64, 128, 128])
         
            x = self.p1(x, res1)
            # torch.Size([16, 64, 256, 256])

            x = self.segmentation_head(x)
            # torch.Size([16, 64, 256, 256])
           
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            # torch.Size([16, 6, 1024, 1024])#batch size, channel, height, width


            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
        
        self.Trans = Attention(512,512)
        
        
        
    def forward(self, x):
        # print(x.shape)#torch.Size([16, 3, 1024, 1024])
        # exit()
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        res4 = self.Trans(res4)
        # print(res1.shape)#torch.Size([16, 64, 256, 256])
        # print(res2.shape)#torch.Size([16, 128, 128, 128])
        # print(res3.shape)#torch.Size([16, 256, 64, 64])
        # print(res4.shape)#torch.Size([16, 512, 32, 32])
        # exit()
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w) 
            # torch.Size([16, 6, 512, 512])
            # print(x.shape)
            # exit()
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            # torch.Size([16, 6, 1024, 1024])
            # print(x.shape)
            # print('1111111111111111111111111111111')
            # exit()
            return x
class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Attention, self).__init__()
        
        self.out = Linear(out_channel, out_channel)
        self.attn_dropout = Dropout(0.2)
        self.proj_dropout = Dropout(0.2)
        self.softmax = Softmax(dim=-1)

        self.conv1 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv2 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv3 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv4 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv5 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv6 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv7 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv8 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.conv9 = Conv2d(in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=1,
                            stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.relu9 = nn.ReLU(inplace=True)
    def transpose_for_scores(self, x):
        # print(x.size()[:-1])#torch.Size([24, 196])
        # print((self.num_attention_heads, self.attention_head_size))#(12, 64)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # print(new_x_shape)#torch.Size([24, 196, 12, 64])
        x = x.view(*new_x_shape)
        # print(x.shape)#torch.Size([24, 196, 12, 64])
        # print(x.permute(0, 2, 1, 3).shape)#torch.Size([24, 12, 196, 64])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # print(hidden_states.shape)#torch.Size([16, 512, 32, 32])
        # exit()

        batch1,channel1,width1,height1 = hidden_states.shape
        index_1 = torch.LongTensor([[31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        index_2 = index_1.repeat(batch1,channel1,width1,1)#batch,channel;width,1

        p1 = torch.gather(hidden_states, dim=3, index=index_2)
        o1 = p1.permute(0, 1,3,2).contiguous()
        p2 = torch.gather(o1, dim=3, index=index_2)
        o2 = p2.permute(0, 1,3,2).contiguous()
        p3 = torch.gather(o2, dim=3, index=index_2)
        o3 = p3.permute(0, 1,3,2).contiguous()

        hidden_states1 = self.relu1(self.conv1(hidden_states))# torch.Size([24, 768, 14, 14])
        hidden_states2 = self.relu2(self.conv2(o1))# torch.Size([24, 768, 14, 14])
        hidden_states3 = self.relu3(self.conv3(o2))# torch.Size([24, 768, 14, 14])
        hidden_states4 = self.relu4(self.conv4(o3))# torch.Size([24, 768, 14, 14])
        hidden_states_11 = hidden_states1 + hidden_states2 + hidden_states3 + hidden_states4
        
        batch2,channel2,width2,height2 = hidden_states.shape
        s = int(width2/2)
        e1 = hidden_states[:, :, 0:s, 0:s]
        e2 = hidden_states[:, :, 0:s, s:width2]
        e3 = hidden_states[:, :, s:width2, 0:s]
        e4 = hidden_states[:, :, s:width2, s:width2]

        index_1 = torch.LongTensor([[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        index_2 = index_1.repeat(batch2,channel2,s,1)

        p1_1 = torch.gather(e1, dim=3, index=index_2)
        o1_1 = p1_1.permute(0,1,3,2).contiguous()
        p2_1 = torch.gather(o1_1, dim=3, index=index_2)
        o2_1 = p2_1.permute(0,1,3,2).contiguous()
        p3_1 = torch.gather(o2_1, dim=3, index=index_2)
        o3_1 = p3_1.permute(0,1,3,2).contiguous()


        p1_2 = torch.gather(e2, dim=3, index=index_2)
        o1_2 = p1_2.permute(0,1,3,2).contiguous()
        p2_2 = torch.gather(o1_2, dim=3, index=index_2)
        o2_2 = p2_2.permute(0,1,3,2).contiguous()
        p3_2 = torch.gather(o2_2, dim=3, index=index_2)
        o3_2 = p3_2.permute(0,1,3,2).contiguous()

        p1_3 = torch.gather(e3, dim=3, index=index_2)
        o1_3 = p1_3.permute(0,1,3,2).contiguous()
        p2_3 = torch.gather(o1_3, dim=3, index=index_2)
        o2_3 = p2_3.permute(0,1,3,2).contiguous()
        p3_3 = torch.gather(o2_3, dim=3, index=index_2)
        o3_3 = p3_3.permute(0,1,3,2).contiguous()

        p1_4 = torch.gather(e4, dim=3, index=index_2)
        o1_4 = p1_4.permute(0,1,3,2).contiguous()
        p2_4 = torch.gather(o1_4, dim=3, index=index_2)
        o2_4 = p2_4.permute(0,1,3,2).contiguous()
        p3_4 = torch.gather(o2_4, dim=3, index=index_2)
        o3_4 = p3_4.permute(0,1,3,2).contiguous()


        t1_1 = torch.cat((o1_1,o1_2),-1)#torch.Size([1, 2, 2, 2])
        t2_1 = torch.cat((o1_3,o1_4),-1)#torch.Size([1, 2, 2, 2])
        t3_1 = torch.cat((t1_1,t2_1),-2)

        t1_2 = torch.cat((o2_1,o2_2),-1)#torch.Size([1, 2, 2, 2])
        t2_2 = torch.cat((o2_3,o2_4),-1)#torch.Size([1, 2, 2, 2])
        t3_2 = torch.cat((t1_2,t2_2),-2)

        t1_3 = torch.cat((o3_1,o3_2),-1)#torch.Size([1, 2, 2, 2])
        t2_3 = torch.cat((o3_3,o3_4),-1)#torch.Size([1, 2, 2, 2])
        t3_3 = torch.cat((t1_3,t2_3),-2)
        
        
        hidden_states5 = self.relu5(self.conv5(hidden_states))# torch.Size([24, 768, 14, 14])
        hidden_states6 = self.relu6(self.conv6(t3_1))# torch.Size([24, 768, 14, 14])
        hidden_states7 = self.relu7(self.conv7(t3_2))# torch.Size([24, 768, 14, 14])
        hidden_states8 = self.relu8(self.conv8(t3_3))
        hidden_states_111 =  hidden_states5 + hidden_states6 + hidden_states7 + hidden_states8# torch.Size([24, 768, 14, 14])
        
        q = hidden_states_11#torch.Size([16, 512, 32, 32]
        k = hidden_states_111#torch.Size([16, 512, 32, 32]
        v = hidden_states_11#torch.Size([16, 512, 32, 32]
        
        att = torch.matmul(q, k)
        attention_probs = self.softmax(att)
        attention = self.attn_dropout(attention_probs)
        out = torch.matmul(attention, v)
        out = self.proj_dropout(self.relu9(self.conv9(out)))
        return out
         
