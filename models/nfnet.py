import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

nonlinearities = {
'silu': lambda x: F.silu(x) / .5595,
'relu': lambda x: F.relu(x) / (0.5 * (1 - 1 / np.pi)) ** 0.5,
'identity': lambda x: x}


class WSConv2d(nn.Conv2d): # weight standardization, affine gain and bias applied conv2d layer
  def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super(WSConv2d, self).__init__(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    # learnable affine gain and initialization
    self.gain = self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
    
  def weight_standardization(self, eps=1e-4):
    fan_in = np.prod(self.weight.shape[1:])
    mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
    var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
    weight = (self.weight - mean) / (var * fan_in + eps) ** 0.5
    if self.gain is not None:
        weight = weight * self.gain
    return weight

  def forward(self, x, eps=1e-4): 
    weight = self.weight_standardization(eps) # to be used in training
    out = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return out
    


class SEblock(nn.Module): # squeeze+excite block
  def __init__(self, in_channel, out_channel, se_ratio = 0.5, hidden_channels=None, activation=F.relu):
    super(SEblock, self).__init__()
    if se_ratio is None:
      hidden_channels = hidden_channels
    else:
      hidden_channels = max(1, int(se_ratio*in_channel))
    self.fc0 = nn.Conv2d(in_channel, hidden_channels, kernel_size=1, bias=True)
    self.fc1 = nn.Conv2d(hidden_channels, in_channel, kernel_size=1, bias=True)
    self.activation = activation

  def forward(self, x):
    out = torch.mean(x, axis=[2, 3], keepdims=True)
    out = self.fc0(out)
    out = self.fc1(self.activation(out))
    out = 2.*torch.sigmoid(out)*x
    return out

class NF_Bottleneck(nn.Module): # basic block for NFNET
  def __init__(self, in_channel, out_channel, se_ratio = 0.5, stride=1, alpha=0.2, beta=1.0, expansion=2.25,
               activation=F.relu, downsample=None):
    super(NF_Bottleneck, self).__init__()
    self.se_ratio = se_ratio # for Squeeze-Excite block
    self.alpha = alpha
    self.beta = beta
    self.activation = activation
    self.stride =stride
    mid_ch = int(in_channel*expansion)

    self.conv0 = WSConv2d(in_channel,mid_ch, 1) # kernel = 1 
    self.conv1 = WSConv2d(mid_ch, mid_ch, 3, stride, 1) # HW reform at transition, padding=1
    self.se = SEblock(mid_ch, mid_ch, se_ratio)

    self.conv2 = WSConv2d(mid_ch, out_channel, 1) # kernel = 1 

    
    self.downsample = downsample
    self.skipinit_gain = nn.Parameter(torch.zeros(() ))

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

class NF_resnet(nn.Module):
  def __init__(self, variant, activation= 'relu', num_classes=10, expansion=2.25, drop_rate=None, stochastic_depth=0.0): 
    super(NF_resnet, self).__init__()
    self.activation = nonlinearities.get(activation)
    self.variant = variant
    #self.width = width
    self.alpha = 0.2
    self.expansion_ratio = expansion
    

    if drop_rate is None: # for dropout
      self.drop_rate = 0.2
    else:
      self.drop_rate = drop_rate
    #self.stochdepth_rate = stochdepth_rate

    self.width = [48, 104, 208, 440]
    self.depth = [2, 4, 7, 7]
    print(self.width)
    #stem
    self.stem = WSConv2d(3, self.width[0], 3, 1, 1) #kernel_size=3, stride =1, padding = 1 

    self.in_planes = self.width[0]
    # Body
    self.layer1 = self._make_layer(self.width[0], self.depth[0])
    self.layer2 = self._make_layer(self.width[1], self.depth[1], stride=2)
    self.layer3 = self._make_layer(self.width[2], self.depth[2], stride=2)
    self.layer4 = self._make_layer(self.width[3], self.depth[3], stride=2) # from 1536 to 1024

    # final
    self.in_planes = int(1280 * self.in_planes // 440)
    self.final_conv = WSConv2d(self.width[3], self.in_planes, 1, 1) # kernel_size= 1
    self.fc = nn.Linear(self.in_planes, num_classes, bias=True)
    nn.init.zeros_(self.fc.weight)

    #dropout
    if self.drop_rate > 0.0:
      self.dropout = nn.Dropout(self.drop_rate)

  def _make_layer(self, planes, blocks, stride=1):
    layers = []
    # initialization
    downsample = nn.Sequential(
        nn.AvgPool2d(2),
        WSConv2d(self.in_planes, planes ,1)
    )

    layers.append(NF_Bottleneck(self.in_planes, planes,  stride=2, alpha=self.alpha, beta= 1., expansion=1, 
                                activation = self.activation, downsample=downsample))
    expected_std = (1.**2 + self.alpha**2)**0.5
    self.in_planes = planes
    for _ in range(1, blocks):
      beta = expected_std
      layers.append(NF_Bottleneck(self.in_planes, planes, stride=1, alpha=self.alpha, beta=beta, expansion=self.expansion_ratio,
                                  activation = self.activation))
      expected_std = (expected_std**2 + self.alpha**2)**0.5
    return nn.Sequential(*layers)

  def forward(self, x):
  
    out = self.stem(x)

  
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.final_conv(out)
    out = self.activation(out)

    pool = torch.mean(out, [2, 3])
    if self.drop_rate > 0.0 and self.training:
      out = self.dropout(pool)

    out = self.fc(pool)
    return out