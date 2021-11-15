from utils.SwitchBlock import *

class Switchable_ResNet50(nn.Module):
  def __init__(self, num_classes=1000):
    super(Switchable_ResNet50, self).__init__()
    self.in_planes = [16, 32, 48, 64] 
    self.conv1 = SlimmableConv2d([3,3,3,3], self.in_planes, kernel_size=3, stride=1, padding=1, bias=False, width_mult_list=width_mult_list) # img with 3 channels
    self.bn1 = SwitchableBatchNorm2d(self.in_planes, width_mult_list=width_mult_list)
    self.relu = nn.ReLU(inplace = True)
    
    self.channels = [  [  int(j*i)   for j in width_mult_list] for i in self.in_planes]
    
    self.layer1 = self._make_layer(self.channels[0], 3) 
    self.layer2 = self._make_layer(self.channels[1], 4, stride=2)
    self.layer3 = self._make_layer(self.channels[2], 6, stride=2)
    self.layer4 = self._make_layer(self.channels[3], 3, stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.dropout = nn.Dropout(0.2)
    self.fc = SlimmableLinear([i*4 for i in self.channels[3]], num_classes, width_mult_list=width_mult_list)
    
  def _make_layer(self, planes, blocks, stride=1): 
    skip_channel = [ 4*i for i in planes]
    downsample = nn.Sequential(
          SlimmableConv2d(self.in_planes, skip_channel, 1, stride, width_mult_list=width_mult_list),
          SwitchableBatchNorm2d(skip_channel, width_mult_list)
      )
    
    layers = []
    layers.append(Bottleneck(self.in_planes, planes ,stride, downsample)) 
    self.in_planes = [i*4 for i in planes] 
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
    if self.training:
      x = self.dropout(x)
    x = self.fc(x)
    
    return x