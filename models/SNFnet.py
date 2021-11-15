from utils.NFblock import *

nonlinearities = {
'silu': lambda x: F.silu(x) / .5595,
'relu': lambda x: F.relu(x) / (0.5 * (1 - 1 / np.pi)) ** 0.5,
'identity': lambda x: x}


class NF_Resnet(nn.Module):
  def __init__(self, activation= 'relu', num_classes=10, expansion=2.25): 
    super(NF_Resnet, self).__init__()
    self.activation = nonlinearities.get(activation)
    
    self.alpha = 0.2
    self.expansion_ratio = expansion

    #self.width = [48, 104, 208, 440] # number of channels list
    self.width = [64, 128, 256, 512] # number of channels list
    self.depth = [1, 3, 6, 6] # number of blocks required    
    self.drop_rate = 0.3 # for dropout

    self.width_list = [[  int(j*i)   for j in width_mult_list] for i in self.width]
    #stem
    self.stem = SlimmableWSConv2d([3,3,3,3], self.width_list[0], 3, 1, 1, width_mult_list=width_mult_list) #kernel_size=3, stride =1, padding = 1 
    self.in_planes = self.width_list[0]
    # Body
    self.layer1 = self._make_layer(self.width_list[0], self.depth[0]) 
    self.layer2 = self._make_layer(self.width_list[1], self.depth[1], stride=2)
    self.layer3 = self._make_layer(self.width_list[2], self.depth[2], stride=2)
    self.layer4 = self._make_layer(self.width_list[3], self.depth[3], stride=2) # 

    # final
    self.in_planes = [int(2*i) for i in self.in_planes]
    self.final_conv = SlimmableWSConv2d(self.width_list[3], self.in_planes, 1, 1, width_mult_list=width_mult_list) # kernel_size= 1
    self.fc = SlimmableLinear(self.in_planes, num_classes, 
                              bias=True, width_mult_list=width_mult_list)
    nn.init.zeros_(self.fc.weight)
    
    # dropout
    if self.drop_rate > 0.0:
      self.dropout = nn.Dropout(self.drop_rate)

  def _make_layer(self, planes, blocks, stride=1):
    layers = []
    # initialization
    downsample = nn.Sequential(
        nn.AvgPool2d(2),
        SlimmableWSConv2d(self.in_planes, planes ,1,  width_mult_list=width_mult_list)
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