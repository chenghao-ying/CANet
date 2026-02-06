import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
# from models.SwinTransformers import SwinTransformer
from models.smt import smt_b

def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )

def conv3x3_bn_gelu(in_planes, out_planes, k=3, s=2, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, 128, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )

class CANet(nn.Module):
    def __init__(self, ):
        super(CANet, self).__init__()

        self.rgb_swin = smt_b()
        self.depth_swin = smt_b()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)

        self.Fusion_1 = Fusion0(1024, 1024)
        self.Fusion_2 = Fusion(512, 512)
        self.Fusion_3 = Fusion(256, 256)
        self.Fusion_4 = Fusion(128, 128)

        self.FA_Block1 = Newblock(256)
        self.FA_Block2 = Newblock(128)
        self.FA_Block3 = Newblock(64)
        self.FA_Block4 = Newblock(32)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 =  nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2
        )


        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2
        )

        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)

        self.dwc4 = conv3x3_bn_relu(64, 32)
        self.dwc3 = conv3x3_bn_relu(128, 64)
        self.dwc2 = conv3x3_bn_relu(256, 128)
        self.dwc1 = conv3x3_bn_relu(512, 256)

        self.dwcon_1 = conv3x3_bn_relu(1024, 512)
        self.dwcon_2 = conv3x3_bn_relu(512, 256)
        self.dwcon_3 = conv3x3_bn_relu(256, 128)
        self.dwcon_4 = conv3x3_bn_relu(128, 64)

        self.conv43 = conv3x3_bn_relu(64, 128, s=2)
        self.conv32 = conv3x3_bn_relu(128, 256, s=2)
        self.conv21 = conv3x3_bn_relu(256, 512, s=2)

        self.convdown = conv3x3_bn_gelu(32, 128)
        self.conv1x1 = nn.Conv2d(256, 128,1)



    def forward(self,x ,d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r4 = rgb_list[0]  # [8, 128, 96, 96]
        r3 = rgb_list[1]  # [8, 256, 48, 48]
        r2 = rgb_list[2]  # [8, 512, 24, 24]
        r1 = rgb_list[3]  # [8, 1024, 12, 12]

        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]

        r4_up = F.interpolate(self.dwc4(r4), size=256, mode='bilinear')
        r3_up = F.interpolate(self.dwc3(r3), size=128, mode='bilinear')
        r2_up = F.interpolate(self.dwc2(r2), size=64, mode='bilinear')
        r1_up = F.interpolate(self.dwc1(r1), size=32, mode='bilinear')
        d3_up = F.interpolate(self.dwc3(d3), size=128, mode='bilinear')
        d2_up = F.interpolate(self.dwc2(r2), size=64, mode='bilinear')
        d1_up = F.interpolate(self.dwc1(r1), size=32, mode='bilinear')

        r1_con = torch.cat((r1, r1), 1)
        r1_con = self.dwcon_1(r1_con)
        d1_con = torch.cat((d1, d1), 1)
        d1_con = self.dwcon_1(d1_con)
        r2_con = torch.cat((r2, r1_up), 1)
        r2_con = self.dwcon_2(r2_con)
        d2_con = torch.cat((d2, d1_up), 1)
        d2_con = self.dwcon_2(d2_con)
        r3_con = torch.cat((r3, r2_up), 1)
        r3_con = self.dwcon_3(r3_con)
        d3_con = torch.cat((d3, d2_up), 1)
        d3_con = self.dwcon_3(d3_con)
        r4_con = torch.cat((r4, r3_up), 1)
        r4_con = self.dwcon_4(r4_con)
        d4_con = torch.cat((d4, d3_up), 1)
        d4_con = self.dwcon_4(d4_con)

        fu1 = self.Fusion_1(r1_con, d1_con, x)  # 1024,12,12
        fu2 = self.Fusion_2(r2_con, d2_con, fu1, x)  # 512,24,24
        fu3 = self.Fusion_3(r3_con, d3_con, fu2, x)  # 256,48,48
        fu4 = self.Fusion_4(r4_con, d4_con, fu3, x)  # 128,96,96

        df_f_1 = self.deconv_layer_1(fu1)  # 512
        df_f_1 = self.FA_Block1(df_f_1, r1_up)

        xc_1_2 = torch.cat((df_f_1, fu2), 1)  # 512 + 512
        df_f_2 = self.deconv_layer_2(xc_1_2)  # 256

        df_f_2 = self.FA_Block2(df_f_2, r2_up)       # 256

        xc_1_3 = torch.cat((df_f_2, fu3), 1)  # 256 + 256
        df_f_3 = self.deconv_layer_3(xc_1_3)  # 128
        df_f_3 = self.FA_Block3(df_f_3, r3_up)       # 128

        xc_1_4 = torch.cat((df_f_3, fu4), 1)  # 128 + 128
        df_f_4 = self.deconv_layer_4(xc_1_4)  # 64
        df_f_4 = self.FA_Block4(df_f_4, r4_up)       # 64

        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=512, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=512, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=512, mode='bilinear')
        return y1,y2,y3,y4

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'])
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'])
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图

def channel_shuffle(x, groups=4):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Fusion0(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Fusion0, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.cbam = CBAM(inp//2)
        self.sa = SpatialAttention(7)
        self.ca = ChannelAttention(inp//2, 16)
        self.layer1 = nn.Sequential(nn.Conv2d(inp, inp//2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(inp//2), act_fn, )
        self.layer2 = nn.Sequential(nn.Conv2d( inp//2, inp//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(inp//2), act_fn, )
        self.layer3 = nn.Sequential(nn.Conv2d( inp//2, inp//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(inp//2), act_fn, )
        self.layer4 = nn.Sequential(nn.Conv2d( inp, inp//2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(inp//2),act_fn, )
        # self.mask = nn.Sequential(BasicConv2d(inp//2, inp//4, 3, padding=1), nn.Conv2d(inp//4, 1, 1))

    def forward(self, rgb, depth, s):
        x = torch.cat((rgb, depth), dim=1)
        x = channel_shuffle(x, 4)
        rgb = self.layer2(rgb)
        depth = self.layer3(depth)
        x1 = self.layer1(x)
        # n, c, h, w = x.size()
        sa_x = self.sa(x1)
        ca_x = self.ca(x1)
        rgb_att = rgb * ca_x
        depth_att = depth * sa_x
        fu = torch.cat((rgb_att, depth_att), dim=1)
        fu = channel_shuffle(fu, 4)
        fu = self.layer4(fu)
        fu_1 = self.cbam(fu)
        fu_2 = fu_1 + x1

        return fu_2

class Fusion(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Fusion, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.cbam = CBAM(inp//2)
        self.sa = SpatialAttention(7)
        self.ca = ChannelAttention(inp//2, 16)
        self.layer1 = nn.Sequential(nn.Conv2d(2*inp, inp//2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(inp//2), act_fn, )
        self.layer2 = nn.Sequential(nn.Conv2d( inp//2, inp//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(inp//2), act_fn, )
        self.layer3 = nn.Sequential(nn.Conv2d( inp//2, inp//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(inp//2), act_fn, )
        self.layer4 = nn.Sequential(nn.Conv2d( inp, inp//2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(inp//2),act_fn, )
        # self.mask = nn.Sequential(BasicConv2d(inp//2, inp//4, 3, padding=1), nn.Conv2d(inp//4, 1, 1))
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, rgb, depth, fu_out, s):
        fu_out2 = self.up2(fu_out)
        x = torch.cat((rgb, depth, fu_out2), dim=1)
        x = channel_shuffle(x, 4)
        rgb = self.layer2(rgb)
        depth = self.layer3(depth)
        x1 = self.layer1(x)
        # n, c, h, w = x.size()
        sa_x = self.sa(x1)
        ca_x = self.ca(x1)
        rgb_att = rgb * ca_x
        depth_att = depth * sa_x
        fu = torch.cat((rgb_att, depth_att), dim=1)
        fu = channel_shuffle(fu, 4)
        fu = self.layer4(fu)
        fu_1 = self.cbam(fu)

        fu_2 = fu_1 + x1

        return fu_2

# without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net
class Newblock(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(Newblock, self).__init__()
        act_fn = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局自适应池化
        self.fc1 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

        self.fc_wight1 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )
        self.fc_wight2 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(nn.Conv2d(ch_in*2, ch_in, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ch_in), act_fn, )
        self.conv2 = nn.Sequential(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ch_in), act_fn, )
        self.sigmoid = nn.Sigmoid()
        self.aspp = ASPP(ch_in, ch_in)
        self.conv3 = nn.Sequential(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ch_in), act_fn, )

    def forward(self, x, y):
        b, c, _, _ = x.size()
        x_a = self.avg_pool(x).view(b, c)
        x_m = self.max_pool(x).view(b, c)
        y_a = self.avg_pool(y).view(b, c)
        y_m = self.max_pool(y).view(b, c)
        x_weight_a = self.fc_wight1(x_a)
        x_weight_m = self.fc_wight1(x_m)
        y_weight_a = self.fc_wight2(y_a)
        y_weight_m = self.fc_wight2(y_m)
        h_weight_a = x_weight_a + y_weight_a  # 1
        h_weight_m = x_weight_m + y_weight_m  # 1
        x_h_a = self.fc1(x_a).view(b, c, 1, 1)
        x_h_m = self.fc2(x_m).view(b, c, 1, 1)
        y_h_a = self.fc3(y_a).view(b, c, 1, 1)
        y_h_m = self.fc4(y_m).view(b, c, 1, 1)
        add_avg = x_h_a + y_h_a
        add_max = x_h_m + y_h_m
        add_avg = self.sigmoid(add_avg)
        add_max = self.sigmoid(add_max)
        x_y = torch.cat((x, y), dim=1)
        x_y = self.conv1(x_y)
        x_y1 = x_y * add_max.expand_as(x_y)
        x_y2 = self.conv2(x_y1)
        x_y3 = x_y2 * add_avg.expand_as(x_y)
        x_y_out =  torch.mul(x_y3, h_weight_a.view(b, 1, 1, 1))  #zhi yong le pinjun
        x_y_out1 = x_y_out + x_y

        aspp_out = self.aspp(x_y_out1)
        out = self.conv3(aspp_out)

        return out