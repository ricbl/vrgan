"""Training a VRGAN

Use this file to train and validate a VRGAN model
"""

import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch.optim as optim
from torch import nn
from . import unet
from .synth_dataset import init_synth_dataloader_original
import math 
from . import opts 
from . import outputs
from . import metrics
import torchvision
from . import regressor
from . import utils_dataset

def init_model(opt):
    net_g = unet.init_model(opt)
    net_r = regressor.init_model(opt)
    return net_g, net_r

def init_optimizer(opt, net_g, net_r):
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.learning_rate_g, betas=(
        0.0, 0.9), weight_decay=0)
    optimizer_r = optim.Adam(net_r.parameters(), lr=opt.learning_rate_r, betas=(
            0.0, 0.9), weight_decay=0)
    return optimizer_g, optimizer_r
    
def train(opt, loader_train, loader_val, net_g, net_r, optim_g, optim_r, output, metric):
    #getting a fixed set of validation images to follow the evolution of the
    # correponding output over several epochs
    (fixed_x, fixed_y), (_,fixed_yprime) = iter(loader_val).next()
    output.log_fixed(fixed_x, fixed_y, fixed_yprime)
    fixed_x = fixed_x.cuda()
    fixed_y = fixed_y.cuda()
    fixed_yprime = fixed_yprime.cuda()
    output.log_delta_x_gt(metrics.get_groundtruth_toy(fixed_yprime, fixed_y))
    
    for epoch_index in range(opt.nepochs):
        if not opt.skip_train:
            net_r.train()
            net_g.train()
            for batch_index, batch_example in enumerate(loader_train):
                (x,y), (_,y_prime) = batch_example
                
                y_prime = y_prime.cuda()
                x = x.cuda()
                y = y.cuda()
                
                #Get all model outputs
                delta_x = net_g(x, y_prime,y)
                x_prime = x + delta_x
                r_x = net_r(x)
                r_xprime = net_r(x_prime)
                
                # Update regressor
                
                optim_r.zero_grad()
                
                # Eq. 2 of paper, L_{Rx}, to make the regressor a good regressor
                # on the original dataset
                l_rx = torch.mean(torch.abs(r_x - y))
                
                # Eq. 4 of paper, L_{Rx'}, to make the regressor being able to
                # ignore changes made by the generator
                l_rxprime = torch.mean(torch.abs(r_xprime - y))
                
                #R* part of Eq. 6
                regressor_loss = opt.lambda_rx * l_rx + opt.lambda_rxprime * l_rxprime

                regressor_loss.backward(retain_graph = True)
                optim_r.step()
                r_xprime = net_r(x_prime)
                #Update generator
                optim_g.zero_grad()
                
                #Eq. 3 on the paper, L_{Gx'}, making the generator to produce a map 
                # that makes the oyutput of the regressor as close as possible to they
                # desired output (y')
                # Here, output regressor is x', i.e. the output of the regressor 
                # model when feeded with the sum of the 
                # original input image (x) summed with the difference map (G(x,y',y))
                l_gxprime = torch.mean(torch.abs(r_xprime - y_prime))
                
                #Eq. 5 in the paper, L_{REG}, penalty on the norm of the
                # difference map to make it just produce the necessary changes 
                l_reg = torch.mean(torch.abs(delta_x))
                
                # G* part of Eq. 6
                gen_loss = opt.lambda_reg*l_reg + opt.lambda_gxprime*l_gxprime
                gen_loss.backward()
                
                optim_g.step()
                
                #save all losses values
                metric.add_value('l_rx', l_rx)
                metric.add_value('l_rxprime', l_rxprime)
                metric.add_value('l_gxprime', l_gxprime)
                metric.add_value('regressor_loss', regressor_loss)
                metric.add_value('gen_loss', gen_loss)
                metric.add_value('l_reg', l_reg)
            output.log_batch(epoch_index, batch_index, metric)
        
        #validation
        with torch.no_grad():
            net_r.eval()
            net_g.eval()
            
            #saving the outputs of the fixed images (always the same images every epoch)
            delta_x = net_g(fixed_x, fixed_yprime, fixed_y)
            output.log_images(epoch_index, net_g, net_r, fixed_x, delta_x)
            
            #iterating through the full validation set
            for batch_index, batch_example in enumerate(loader_val):
                #get validation data
                (x,y),(_,yprime)= batch_example
                yprime = yprime.cuda()
                x = x.cuda()
                y = y.cuda()
                
                #get validation output
                delta_x = net_g(x, yprime, y)
                
                #calculate normalized cross correlation
                metric.add_ncc(yprime, y, delta_x)
                
            net_r.train()
            net_g.train()
        output.log_added_values(epoch_index, metric)
    
def main():
    torch.autograd.set_detect_anomaly(True)
    
    #get user options/configurations
    opt = opts.get_opt()
    
    #set cuda visible devices if user specified a value
    if opt.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    
    #load class to save metrics, iamges and models to disk
    output = outputs.Outputs(opt)
    output.save_run_state(os.path.dirname(__file__))
    
    #load class to store metrics and losses values
    metric = metrics.Metrics()
    
    #load synthetic dataset
    healthy_dataloader_train, anomaly_dataloader_train = init_synth_dataloader_original(
        opt.folder_toy_dataset, opt.batch_size, mode='train')
    
    loader_train = utils_dataset.IteratorLoaderDifferentSizes(healthy_dataloader_train,anomaly_dataloader_train,200)
    healthy_dataloader_val, anomaly_dataloader_val = init_synth_dataloader_original(
        opt.folder_toy_dataset, opt.batch_size, mode='val')
    loader_val = utils_dataset.IteratorLoaderDifferentSizes(healthy_dataloader_val,anomaly_dataloader_val, len(anomaly_dataloader_val.dataset))
    
    net_g, net_r = init_model(opt)

    optim_g, optim_r = init_optimizer(opt, net_g=net_g, net_r=net_r)
    
    train(opt,
          loader_train, loader_val,
          net_g=net_g, net_r=net_r, optim_g=optim_g, optim_r=optim_r, output=output, metric=metric)

if __name__ == '__main__':
    main()
