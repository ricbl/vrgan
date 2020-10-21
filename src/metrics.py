"""VRGAN metrics

File that contain functions for calculating the normalized cross-correlation 
between two images and a class Metrics for storing losses and metrics during 
mini batch inferences, so that you can get an epoch summary after the epoch 
is complete.
"""

import collections
import torch
from . import synth_dataset

def normalized_cross_correlation(a,v): 
    a = a.squeeze(1)
    v = v.squeeze(1)  
    norm_std = torch.std(a.view([a.size(0),-1]), dim = 1)*torch.std(v.view([v.size(0),-1]), dim = 1)
    step1a = (a - torch.mean(a.view([a.size(0),-1]), dim = 1).unsqueeze(1).unsqueeze(2))
    step1v = (v - torch.mean(v.view([v.size(0),-1]), dim = 1).unsqueeze(1).unsqueeze(2))
    step2 = torch.sum((step1a*step1v).view([a.size(0),-1]), dim = 1)
    step3 = step2/norm_std
    step3 = step3/torch.prod(torch.tensor(a.size()[-2:]))
    return step3
    
def get_groundtruth_toy(pft_desired, pft_true):
    I0 = torch.zeros([pft_desired.size(0),1, 224,224])
    I1 = torch.zeros([pft_desired.size(0),1, 224,224])
    for i in range(pft_desired.size(0)):
        I0[i,0,:,:] = torch.tensor(synth_dataset.get_clean_square(pft_true[i], 224))
        I1[i,0,:,:] = torch.tensor(synth_dataset.get_clean_square(pft_desired[i], 224))
    im_diff = ((I1-I0))
    return im_diff.cuda().float()

class Metrics():
    def __init__(self):
        self.values = collections.defaultdict(list)
    
    def add_ncc(self, pft_desired, pft_true, delta_x):
        pft_desired = pft_desired.detach()
        pft_true = pft_true.detach()
        delta_x = delta_x.detach()
        gt_toy = get_groundtruth_toy(pft_desired, pft_true)
        self.add_list('ncc', normalized_cross_correlation(gt_toy, delta_x))
    
    def add_list(self, key, value):
        value = value.detach().cpu().tolist()
        self.values[key] += value
        
    def add_value(self, key, value):
        value = value.detach().cpu()
        self.values[key].append( value)
    
    def calculate_average(self):
        self.average = {}
        for key, element in self.values.items():
            n_values = len(element)
            sum_values = sum(element)
            self.average[key] = sum_values/float(n_values)
        self.values = collections.defaultdict(list)
    
    def get_average(self):
        self.calculate_average()
        return self.average
        
    def get_keys(self):
        return self.values.keys()
        
    def get_last_added_value(self,key):
        return self.values[key][-1]