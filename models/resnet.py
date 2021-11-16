import torch
import torch.nn as nn

class Bottleneck(nn.Module):
  def __init__(self, inp, outp, stride=1, downsample=None): # downsample at the beginning of the layer
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inp, outp, 1, 1, bias=False) # kernel_size =1, stride=1
    self.bn1 = nn.BatchNorm2d(outp)
    self.conv2 = nn.Conv2d(outp, outp, 3, stride, 1, bias=False) # kernel_size =3, stride=2, padding=1
    self.bn2 = nn.BatchNorm2d(outp)
    self.conv3 = nn.Conv2d(outp, outp*4, 1)
    self.bn3 = nn.BatchNorm2d(outp*4)
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

#define ResNet50
class ResNet50(nn.Module):
  def __init__(self, num_classes=1000):
    super(ResNet50, self).__init__()
    #self.num_classes = 1000 # imageNet
    self.in_planes = 64 # input마다 달리함
    self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(self.in_planes)
    self.relu = nn.ReLU(inplace = True)
    self.layer1 = self._make_layer(64, 3) 
    self.layer2 = self._make_layer(128, 4, stride=2)
    self.layer3 = self._make_layer(256, 6, stride=2)
    self.layer4 = self._make_layer(512, 3, stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(2048, num_classes)
    
  def _make_layer(self, planes, blocks, stride=1): 
    downsample = nn.Sequential( # for skip connection
          nn.Conv2d(self.in_planes, planes*4,1,stride), # kernel_size = 1
          nn.BatchNorm2d(planes*4)
      )
    layers = []
    layers.append(Bottleneck(self.in_planes, planes ,stride, downsample)) 
    self.in_planes = planes * 4 
    for _ in range(1, blocks):
        layers.append(Bottleneck(self.in_planes, planes)) 
    return nn.Sequential(*layers)
    

  def forward(self, x: torch.Tensor):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    
    x = self.avgpool(x)
    x = torch.flatten(x,1)
    x = self.fc(x)
    
    return x