import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

width_mult_list = [0.25, 0.50, 0.75, 1.0]

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_channels_list, width_mult_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_channels_list = num_channels_list
        bn_group = []
        for i in num_channels_list:
          bn_group.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bn_group)
        self.width_mult_list = width_mult_list
        self.width_mult = max(width_mult_list) # 시작점

    def forward(self, x):
        idx = self.width_mult_list.index(self.width_mult)
        out = self.bn[idx](x)
        return out

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list, kernel_size, stride=1, padding=0,
                dilation = 1, groups_list = [1], bias=True, width_mult_list=[1]):
        super(SlimmableConv2d, self).__init__( 
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.width_mult_list = width_mult_list
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]: 
             self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list) 

    def forward(self, input): # in test

        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d( 
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y

class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features, bias=True, width_mult_list= [1]): 
        super(SlimmableLinear, self).__init__(
            max(in_features_list), out_features, bias=bias)
        self.out_features = out_features
        self.width_mult_list = width_mult_list
        self.in_features_list = in_features_list
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        out = nn.functional.linear(input, weight, bias)            
        return out 

class Bottleneck(nn.Module):
  def __init__(self, inp, outp, stride=1, downsample=None): # downsample at the beginning of the layer
    super(Bottleneck, self).__init__()
    self.conv1 = SlimmableConv2d(inp, outp, 1, 1, bias=False, width_mult_list=width_mult_list) 
    self.bn1 = SwitchableBatchNorm2d(outp, width_mult_list)

    self.conv2 = SlimmableConv2d(outp, outp, 3, stride, 1, bias=False, width_mult_list=width_mult_list) # kernel_size =3, stride=2, padding=1
    self.bn2 = SwitchableBatchNorm2d(outp, width_mult_list)

    self.conv3 = SlimmableConv2d(outp, [4*i for i in outp], 1, width_mult_list=width_mult_list)
    self.bn3 = SwitchableBatchNorm2d([4*i for i in outp], width_mult_list)
    
    self.relu = nn.ReLU(inplace = True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    return out    