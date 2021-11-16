import torch
import torchvision
import torchvision.transforms as transforms
from utils.utils import *

width_mult_list = [0.25, 0.50, 0.75, 1.0]


def train(epoch, net, params, train_loader, device, optimizer, criterion):
  # set random seed
  #set_random_seed()
  print('\n Current train epoch : %d ' % epoch)
  net.train()
  # learning rate difference
  max_width = max(width_mult_list)
  min_width = min(width_mult_list)

  #for each width_mult
  correct = []
  total = 0
  max_accuracy = 0
  train_loss = 0

  #set for each width_mult
  for _ in range(len(width_mult_list)):
    correct.append(0)
  
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    total += targets.size(0) # 계산 data 수 
    
    optimizer.zero_grad()
    tic()
    for i, width_mult in enumerate(sorted(width_mult_list, reverse=True)):
      net.apply(lambda m: setattr(m, 'width_mult', width_mult )) # max 부터
      outputs = net(inputs) 
      loss =  torch.mean(criterion(outputs, targets)) # accumulate loss and calculate mean descent
      loss.backward() # accumulate gradients and perform backward later
      
      train_loss += loss.item()/4 # get average value among widths
      _, predicted = outputs.max(1)
      
      correct[i] += (predicted == targets).sum().item()
    optimizer.step()
    if batch_idx == 0:
      params['time_count'] += toc()
    else:
      params['time_count'] += toc(False)

    if batch_idx % 100 == 0:
      print('\nCurrent batch : %d' % batch_idx)
      for i in range(4):
        print('  Current train accuracy for width_mult [{}] : {}%'.format(width_mult_list[3-i], 100.*correct[i]/total) ) 
      print(' Current Average train loss ', train_loss)
  print('\n')
  for i in range(4):
    print('Total train accuracy for width_mult [{}] : {}%'.format(width_mult_list[3-i], 100.*correct[i]/total) )
    params['train_accuracy_width{}'.format(4-i)].append(correct[i]/total)
  print('Total training loss ', train_loss)
  params['train_loss'].append(train_loss)

def test(epoch, net, params, test_loader, device, optimizer, criterion, path, file_name):
  print('\n Current epoch : %d ' % epoch)
  net.eval()

  correct = []
  test_loss=0
  total = 0
  max_accuracy = 0

  for _ in range(len(width_mult_list)):
    correct.append(0)

  for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    total += targets.size(0) # 계산 data 수 

    for i, width_mult in enumerate(sorted(width_mult_list, reverse=True)):
      net.apply(lambda m: setattr(m, 'width_mult', width_mult )) # max 부터
      outputs = net(inputs) 
      loss =  torch.mean(criterion(outputs, targets)) # accumulate loss and calculate mean descent
      test_loss += loss.item()/4 # get average value among widths
      _, predicted = outputs.max(1)
      
      correct[i] += (predicted == targets).sum().item()

  for i in range(4):
    print('Total test accuracy for width_mult [{}] : {}%'.format(width_mult_list[3-i], 100.*correct[i]/total) ) 
    params['test_accuracy_width{}'.format(4-i)].append(correct[i]/total)
  print('Total test loss ', test_loss)
  params['test_loss'].append(test_loss)

  if max_accuracy <= 100.*correct[i]/total: # 최대 지점 저장, redundant
    max_accuracy = 100.*correct[i]/total
    state = {
      'net': net.state_dict()
    }
    torch.save(state, path + file_name)
    print('Model Saved!')

