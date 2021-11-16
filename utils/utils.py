import numpy as np
import torch
import time
import json

params = {'time_count': 0.0, 'train_accuracy_width1': [], 'train_accuracy_width2': [], 'train_accuracy_width3': [],
         'train_accuracy_width4': [], 'test_accuracy_width1': [], 'test_accuracy_width2': [], 'test_accuracy_width3': [],
         'test_accuracy_width4': [], 'train_loss' : [], 'test_loss' : []}

# for measuring time lapse
def timeGenerator():
  ti = 0.
  tf = time.time()
  while True:
    ti = tf
    tf = time.time()
    yield tf-ti
    
TicToc = timeGenerator()

def toc(tempBool=True):
  tempTimeInterval = next(TicToc)
  if tempBool:
    return tempTimeInterval
  else:
    return 0.

def tic():
  toc(False)


# setting random seed for training
def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# write out as json for plots
def writeToJson(saveTo, params, name):
    params_file = open(saveTo+name, "w")
    json.dump(params, params_file)
    params_file.close()

