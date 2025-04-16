# torch libraries
import torch
import torch.nn as nn
# customized libraries
from Net.PVTv2 import *
from Net.p2t import Encoder_p2t_base
from Net.p2t import Encoder_p2t_tiny
from Net.p2t import Encoder_p2t_small
from Net.p2t import Encoder_p2t_large
import torch.nn.functional as F
from math import log
import timm
# from Net.Res2Net import res2net50_v1b_26w_4s


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)





class MultiScaleFusionModel(nn.Module):
    def __init__(self, in_channel):
        super(MultiScaleFusionModel,self).__init__()
        self.conv_l = ConvBR(in_channel, in_channel, 3, stride=1,padding=1)
        self.conv_m = ConvBR(in_channel, in_channel, 3, stride=1,padding=1)
        self.conv_s = ConvBR(in_channel, in_channel, 3, stride=1,padding=1)
        self.conv1 = ConvBR(in_channel*3, in_channel*3, 3, stride=1,padding=1)
        self.conv2 = ConvBR(in_channel*3, in_channel, 3, stride=1,padding=1)
        self.conv3 = ConvBR(in_channel, 1, 3, stride=1,padding=1)

        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(192, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, 1),
            nn.Softmax(dim=1),
        )
    def forward(self, x_l,x_m,x_s,image_size):
       
        x_l = self.conv_l(x_l) 
        x_l = F.interpolate(x_l, image_size, mode='bilinear', align_corners=False)
        x_l = self.conv_l(x_l) 


        x_m = self.conv_m(x_m)
        x_m = F.interpolate(x_m, image_size, mode='bilinear', align_corners=False)
        x_m = self.conv_l(x_m) 

 
        x_s = self.conv_s(x_s)  
        x_s = F.interpolate(x_s, image_size, mode='bilinear', align_corners=False)
        x_s = self.conv_s(x_s)  


        out_att = self.att(torch.cat((x_l, x_m, x_s), dim=1)) 
        out = self.conv1(torch.cat((x_l, x_m, x_s), dim=1)) 
        out = out * out_att 
        out = self.conv2(out) 
        pred = self.conv3(out)
        return out, pred


class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        B, C, H, W = x.size()
        q_x = x.view(B, C, -1) 
        k_x = x.view(B, C, -1)
        mask = mask.view(B, 1, -1)
        q_x = q_x * mask
        k_x = k_x * mask
        k_x = k_x.permute(0, 2, 1) 

        energy = torch.bmm(q_x, k_x) 
        attention = self.softmax(energy) 
        v_x = x.view(B, C, -1) #

        out = torch.bmm(attention, v_x)
        out = out.view(B, C, H, W)

        out = self.gamma * out + x 
        return out

class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask):
        B, C, H, W = x.size()
        q_x = self.query_conv(x).view(B, -1, W * H)
        k_x = self.key_conv(x).view(B, -1, W * H)
        mask = mask.view(B, -1, W * H)
        q_x = q_x * mask
        k_x = k_x * mask
        q_x = q_x.permute(0, 2, 1)
        energy = torch.bmm(q_x, k_x) 
        attention = self.softmax(energy) 
        v_x = self.value_conv(x).view(B, -1, W * H) 

        out = torch.bmm(v_x, attention.permute(0, 2, 1)) 
        out = out.view(B, C, H, W) 

        out = self.gamma * out + x 
        return out


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)

        self.conv1 = ConvBR(channel, channel, 3, stride=1,padding=1)
        self.conv2 = ConvBR(channel, channel, 3, stride=1,padding=1)
        self.conv3 = ConvBR(channel * 2, channel, 3, stride=1,padding=1)
        self.conv4 = ConvBR(channel, channel, 3, stride=1,padding=1)
    def forward(self, cur_x, h_x, mask):
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)

        out_1 = self.cab(cur_x, mask_d)
        out_2 = self.sab(cur_x, mask_d)
        out_1 = self.conv1( h_x * out_1)
        out_2 = self.conv2( h_x * out_2)
        out = self.conv3(torch.cat((out_1, out_2), dim=1))
        out = self.conv4(out + h_x)
        return out


class MyNet(nn.Module):
    def __init__(self, channel=32, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]):
        super(MyNet, self).__init__()
        channel = channel
        self.model_arc = arc
        if arc == 'PVTv2-B0':
            print('--> using PVTv2-B0 right now')
            self.context_encoder = pvt_v2_b0(pretrained=True)  
            in_channel_list = [64, 160, 256]
        elif arc == 'PVTv2-B1':
            print('--> using PVTv2-B1 right now')
            self.context_encoder = pvt_v2_b1(pretrained=True)  
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B2':
            print('--> using PVTv2-B2 right now')
            self.context_encoder = pvt_v2_b2(pretrained=True)  
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B2-li':
            print('--> using PVTv2-B2-li right now')
            self.context_encoder = pvt_v2_b2_li(pretrained=True)  
            in_channel_list = [128, 320, 512]

        elif arc == 'PVTv2-B4':
            print('--> using PVTv2-B4 right now')
            self.context_encoder = pvt_v2_b4(pretrained=True)  
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B5':
            print('--> using PVTv2-B5 right now')
            self.context_encoder = pvt_v2_b5(pretrained=True)  
            in_channel_list = [128, 320, 512]
        elif arc == 'P2T-base':
            print('--> using P2T-base right now')
            self.context_encoder = Encoder_p2t_base()  
            in_channel_list = [128, 320, 512]
        elif arc == 'P2T-small':
            print('--> using P2T-small right now')
            self.context_encoder = Encoder_p2t_small()  
            in_channel_list = [128, 320, 512]
        elif arc == 'P2T-tiny':
            print('--> using P2T-tiny right now')
            self.context_encoder = Encoder_p2t_tiny() 
            in_channel_list = [96, 240, 384]
        elif arc == 'P2T-large':
            print('--> using P2T-large right now')
            self.context_encoder = Encoder_p2t_large()  
            in_channel_list = [128, 320, 640]
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))

        self.dr2 = DimensionalReduction(in_channel=channel, out_channel=64)
        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=64)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=64)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=64)
        self.CNN_encode1 = ConvBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.CNN_encode2 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.CNN_encode3 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.CNN_encode4 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)

        self.upsample15 = nn.Upsample(scale_factor=1.5, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.msfm = MultiScaleFusionModel(64)
        self.decode4 = Decoder(64)
        self.decode3 = Decoder(64)
        self.decode2 = Decoder(64)
        self.decode1 = Decoder(64)
        self.pred4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    def forward(self, x):

        if self.model_arc == 'PVTv2-B4' or self.model_arc == 'PVTv2-B5' or self.model_arc == 'PVTv2-B0':
            endpoints = self.context_encoder.extract_endpoints(x)
            x1 = endpoints['reduction_2']  
            x2 = endpoints['reduction_3']  
            x3 = endpoints['reduction_4']  
            x4 = endpoints['reduction_5']  
        elif self.model_arc == 'P2T-base' or self.model_arc == 'P2T-small' or self.model_arc == 'P2T-tiny' or self.model_arc == 'P2T-large':
            
            x4, x3, x2, x1 = self.context_encoder(x)

        shape = x.size()[2:]
        xr1 = self.dr2(x1) 
        xr2 = self.dr3(x2) 
        xr3 = self.dr4(x3)
        xr4 = self.dr5(x4)

       
        o_x = x 
        o_x15 = self.upsample15(x) 
        o_x05 = self.upsample05(x) 

        x_l_1 = self.CNN_encode1(o_x15)
        x_l_2 = self.CNN_encode2(x_l_1)
        x_l_3 = self.CNN_encode3(x_l_2)
        x_l_4 = self.CNN_encode4(x_l_3)

        x_m_1 = self.CNN_encode1(o_x)
        x_m_2 = self.CNN_encode2(x_m_1)
        x_m_3 = self.CNN_encode3(x_m_2)
        x_m_4 = self.CNN_encode4(x_m_3)

        x_s_1 = self.CNN_encode1(o_x05)
        x_s_2 = self.CNN_encode2(x_s_1)
        x_s_3 = self.CNN_encode3(x_s_2)
        x_s_4 = self.CNN_encode4(x_s_3)


        out_cnn, cnn_pred = self.msfm(x_l_4, x_m_4, x_s_4, xr4.shape[2:]) 

        # new decoder
        d4 = self.decode4(out_cnn, xr4, cnn_pred)
        d4 = F.interpolate(d4, size=xr3.size()[2:], mode='bilinear')
        p4 = self.pred4(d4)

        d3 = self.decode3(d4, xr3, p4)
        d3 = F.interpolate(d3, size=xr2.size()[2:], mode='bilinear')
        p3 = self.pred3(d3)

        d2 = self.decode2(d3, xr2, p3)
        d2 = F.interpolate(d2, size=xr1.size()[2:], mode='bilinear')
        p2 = self.pred2(d2)

        d1 = self.decode1(d2, xr1, p2)
        p1 = self.pred1(d1)

        p1 = F.interpolate(p1, size=shape, mode='bilinear')
        # p2 = F.interpolate(p2, size=shape, mode='bilinear')
        # p3 = F.interpolate(p3, size=shape, mode='bilinear')
        # p4 = F.interpolate(p4, size=shape, mode='bilinear')
        p_cnn = F.interpolate(cnn_pred, size=shape, mode='bilinear')

        return p1


if __name__ == '__main__':
    net = timm.create_model(model_name="resnet18", pretrained=False, in_chans=3, features_only=True)
    print(net.default_cfg)
    # inputs = torch.randn(1, 3, 352, 352)
    # outs = net(inputs)
    # print(outs[0].shape)