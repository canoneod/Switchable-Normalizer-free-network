# building blocks for Normalizer-Free networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

width_mult_list = [0.25, 0.50, 0.75, 1.0]

class SlimmableWSConv2d(nn.Conv2d):
  def __init__(self, in_channels_list, out_channels_list, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, width_mult_list=[1]):
    super(SlimmableWSConv2d, self).__init__(max(in_channels_list), max(out_channels_list), kernel_size, 
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.in_channels_list = in_channels_list
    self.out_channels_list = out_channels_list
    self.width_mult_list = width_mult_list
    self.width_mult = max(width_mult_list)
    self.fan_in_list = [np.prod(self.weight[:out_channels_list[i],:in_channels_list[i],:,:].shape[1:]) for i in range(len(width_mult_list))]
    self.gain_list = nn.ParameterList(nn.Parameter(torch.ones(self.out_channels_list[i], 1, 1, 1)) for i in range(len(width_mult_list)))

  def weight_standardization(self, eps=1e-4):
    idx = self.width_mult_list.index(self.width_mult) 
    curr_out_ch = self.out_channels_list[idx]
    curr_in_ch = self.in_channels_list[idx]
    
    weight = self.weight[:curr_out_ch,:curr_in_ch,:,:]
    fan_in = self.fan_in_list[idx]
    mean = torch.mean(weight, axis=[1, 2, 3], keepdims=True)
    var = torch.var(weight, axis=[1, 2, 3], keepdims=True)
    weight = (weight - mean) / (var * fan_in + eps) ** 0.5
    return weight

  def forward(self, x, eps=1e-4): 
    idx = self.width_mult_list.index(self.width_mult)
    if self.bias is not None:
      bias = self.bias[:self.out_channels_list[idx]]
    weight = self.weight_standardization(eps)*self.gain_list[idx]
    out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    return out

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

class SlimmableConv2d(nn.Conv2d): # SE Block에서 사용
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

class SwitchableSEblock(nn.Module): # squeeze+excite block
  def __init__(self, in_channels_list, out_channels_list, se_ratio = 0.5, hidden_channels=None, activation=F.relu):
    super(SwitchableSEblock, self).__init__()
    if se_ratio is None:
      hidden_channels = hidden_channels
    else:
      hidden_channels = [max(1, int(se_ratio*i)) for i in in_channels_list]
    self.fc0 = SlimmableConv2d(in_channels_list, hidden_channels, kernel_size=1, bias=True, width_mult_list=width_mult_list)
    self.fc1 = SlimmableConv2d(hidden_channels, out_channels_list, kernel_size=1, bias=True, width_mult_list=width_mult_list)
    self.activation = activation

  def forward(self, x):
    out = torch.mean(x, axis=[2, 3], keepdims=True)
    out = self.fc0(out)
    out = self.fc1(self.activation(out))
    out = 2.*torch.sigmoid(out)*x
    return out
  
class NF_Bottleneck(nn.Module): # building block for NFNET
  def __init__(self, in_channels_list, out_channels_list, se_ratio = 0.5, stride=1, alpha=0.2, beta=1.0, expansion=2.25,
               activation=F.relu, downsample=None):
    super(NF_Bottleneck, self).__init__()
    self.se_ratio = se_ratio # for Squeeze-Excite block
    self.alpha = alpha
    self.beta = beta
    self.activation = activation
    self.stride =stride
    
    mid_ch = [int(i*expansion) for i in in_channels_list]

    self.conv0 = SlimmableWSConv2d(in_channels_list,mid_ch, 1, width_mult_list=width_mult_list) # kernel = 1 
    self.conv1 = SlimmableWSConv2d(mid_ch, mid_ch, 3, stride, 1, width_mult_list=width_mult_list) # HW reform at transition, padding=1
    self.se = SwitchableSEblock(mid_ch, mid_ch, se_ratio)

    self.conv2 = SlimmableWSConv2d(mid_ch, out_channels_list, 1, width_mult_list=width_mult_list) # kernel = 1 

    self.downsample = downsample
    self.skipinit_gain = nn.Parameter(torch.zeros(()))
    self.width_mult = max(width_mult_list)

  def forward(self, x):
    skip = x
    out = self.activation(x) / self.beta
    if self.downsample is not None:
      skip = self.downsample(out)

    out = self.conv0(out)
    out = self.activation(out)

    out = self.conv1(out)
    out = self.activation(out)

    out = self.se(out)
    out = self.activation(out)
    out = self.conv2(out)

    return out*self.alpha*self.skipinit_gain + skip

