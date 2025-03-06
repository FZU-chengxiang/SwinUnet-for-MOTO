from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from SWA_Block import BasicLayer
from PC_block import BasicStage


# This is to optimize the 80x80 dataset in the Stage_2

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,depth, kernel_size,stride,padding):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            BasicStage(dim = out_ch,depth = depth,n_div=4))
        
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size = 6,stride = 2, padding = 2, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class SWPU_Net(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 1,filters = 64,swin_depth = [2,2,2],swin_heads = [4,8,16],pconv_depth = [1,3,6,12,24]):
        super(SWPU_Net, self).__init__()

        n1 = filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.preConv = nn.Sequential(
            nn.Conv2d(in_ch,filters[0],kernel_size = 1,stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace = True),
            BasicStage(dim = filters[0],depth = pconv_depth[0],n_div = 4)
        )
        
        self.Conv1 = conv_block(filters[0], filters[1],pconv_depth[1],3,1,1)
        self.Conv2 = conv_block(filters[1], filters[2],pconv_depth[2],3,1,1)
        self.Conv3 = conv_block(filters[2], filters[3],pconv_depth[3],3,1,1)
        self.Conv4 = conv_block(filters[3], filters[3],pconv_depth[4],7,1,3)
        
        self.BasicLayer1=BasicLayer(dim=filters[0],input_resolution=(40,80),depth=swin_depth[0], num_heads=swin_heads[0], window_size=5)
        self.BasicLayer2=BasicLayer(dim=filters[1],input_resolution=(20,40),depth=swin_depth[1], num_heads=swin_heads[1], window_size=5)
        self.BasicLayer3=BasicLayer(dim=filters[2],input_resolution=(10,20),depth=swin_depth[2], num_heads=swin_heads[2], window_size=5)
        
        self.BasicStage4=BasicStage(dim=filters[3],depth=pconv_depth[3],n_div=4)
        self.BasicStage3=BasicStage(dim=filters[2],depth=pconv_depth[2],n_div=4)
        self.BasicStage2=BasicStage(dim=filters[1],depth=pconv_depth[1],n_div=4)

        self.Up4 = up_conv(filters[3], filters[2])

        self.Up3 = up_conv(filters[3], filters[1])

        self.Up2 = up_conv(filters[2], filters[0])

        self.Conv = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            BasicStage(dim=filters[0],depth=pconv_depth[0],n_div=4),
            nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):

        x= self.preConv(x)
        x_to=self.BasicLayer1(x)
        e1 = self.Conv1(x)
        
        e2= self.Maxpool1(e1)
        e2_to=self.BasicLayer2(e2)
        e2 = self.Conv2(e2_to)

        e3 = self.Maxpool2(e2)
        e3_to=self.BasicLayer3(e3)
        e3 = self.Conv3(e3)

        e4_to = self.Maxpool3(e3)
        e4 = self.Conv4(e4_to)
        
        d4=self.Up4(e4)
        d4 = torch.cat((e3_to, d4), dim=1)
        d4=self.BasicStage4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((e2_to, d3), dim=1)
        d3=self.BasicStage3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x_to, d2), dim=1)
        out = self.Conv(d2)

        return out