"""Training a VRGAN

Use this file to train and validate a VRGAN model
"""

import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch.optim as optim
from torch import nn
from unet import UNet
from synth_dataset import init_synth_dataloader_original
import math 
import opts 
import outputs
import metrics
import torchvision

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        pass
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        pass
        torch.nn.init.normal_(m.weight, mean=0.1, std=0.01)
        m.bias.data.fill_(0)

def init_model(opt):
    net_g = UNet(nf=64)
    if opt.load_checkpoint_g is not None:
        net_g.load_state_dict(torch.load(opt.load_checkpoint_g))
    else:
        net_g.apply(weights_init)
    
    net_r = torchvision.models.resnet18(pretrained = True) 
    net_r.fc = torch.nn.Linear(in_features = 512, out_features = 1)
    if opt.load_checkpoint_r is not None:
        net_r.load_state_dict(torch.load(opt.load_checkpoint_r))
    return net_g, net_r

def init_optimizer(opt, net_g, net_r):
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.learning_rate_g, betas=(
        0.0, 0.9), weight_decay=0)
    optimizer_r = optim.Adam(net_r.parameters(), lr=opt.learning_rate_r, betas=(
            0.0, 0.9), weight_decay=0)
    return optimizer_g, optimizer_r

class BatchNormalizeTensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        to_return = (tensor-self.mean)/self.std
        return to_return

#normalize input images with imagenet normalization values before using them 
# with the imagenet pre-trained regressor
def preprocess_input_resnet(x):
    return BatchNormalizeTensor(torch.FloatTensor([0.485, 0.456, 0.406]).cuda().view([1,3,1,1]), 
            torch.FloatTensor([0.229, 0.224, 0.225]).cuda().view([1,3,1,1]))((x).expand([-1,3,-1,-1]))

#normalize PFT values y before concatenating them to the unet
def preprocess_pft_values_for_generator(x):
    return BatchNormalizeTensor(torch.FloatTensor([0.7]).cuda(), torch.FloatTensor([0.2]).cuda())(x)
    
def train(opt, healthy_dataloader, anomaly_dataloader, healthy_dataloader_val,anomaly_dataloader_val, net_g, net_r, optim_g, optim_r, output, metric):
    #getting a fixed set of validation images to follow the evolution of the
    # correponding output over several epochs
    fixed_x, fixed_y = iter(healthy_dataloader_val).next()
    _ ,fixed_yprime = iter(anomaly_dataloader_val).next()
    output.log_fixed(fixed_x, fixed_y, fixed_yprime)
    fixed_x = fixed_x.cuda()
    fixed_y = fixed_y.cuda()
    fixed_yprime = fixed_yprime.cuda()
    output.log_delta_x_gt(metrics.get_groundtruth_toy(fixed_yprime, fixed_y))
    
    for epoch_index in range(opt.nepochs):
        data_iter = iter(healthy_dataloader)
        anomaly_data_iter = iter(anomaly_dataloader)
        i = 0
        k = 0
        if not opt.skip_train:
            net_r.train()
            net_g.train()
            
            #following Baumgartner et al. (2018), turning off batch normalization 
            # on the regressor (equivalent of the critic)
            for module in net_r.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d) or \
                isinstance(module, torch.nn.modules.BatchNorm2d) or \
                isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
                    
            total_iters_per_epoch = 200
            
            for batch_index in range(total_iters_per_epoch):
                # get data from two dataloaders
                
                if i >= len(healthy_dataloader):
                    data_iter = iter(healthy_dataloader)
                    i = 0
                _, y_prime = data_iter.next()
                y_prime = y_prime.cuda()
                i += 1

                if k >= len(anomaly_dataloader):
                    anomaly_data_iter = iter(anomaly_dataloader)
                    k = 0

                x, y = anomaly_data_iter.next()
                k += 1
                
                x = x.cuda()
                y = y.cuda()
                
                #Get all model outputs
                delta_x = net_g(x, preprocess_pft_values_for_generator(y_prime),preprocess_pft_values_for_generator(y))
                x_prime = x + delta_x
                r_x = net_r(preprocess_input_resnet(x))#[:,0]#, [], None, 0)[:,0]
                r_xprime = net_r(preprocess_input_resnet(x_prime))#[:,0]#, [], None, 0)[:,0]
                
                # Update regressor
                
                optim_r.zero_grad()
                
                # Eq. 2 of paper, L_{Rx}, to make the regressor a good regressor
                # on the original dataset
                l_rx = torch.abs(r_x - y).mean()
                
                # Eq. 4 of paper, L_{Rx'}, to make the regressor being able to
                # ignore changes made by the generator
                l_rxprime = torch.abs(r_xprime - y).mean()
                
                #R* part of Eq. 6
                regressor_loss = opt.lambda_rx * l_rx + opt.lambda_rxprime * l_rxprime

                regressor_loss.backward(retain_graph = True)
                optim_r.step()
                
                #Update generator
                
                optim_g.zero_grad()
                
                #Eq. 3 on the paper, L_{Gx'}, making the generator to produce a map 
                # that makes the oyutput of the regressor as close as possible to they
                # desired output (y')
                # Here, output regressor is x', i.e. the output of the regressor 
                # model when feeded with the sum of the 
                # original input image (x) summed with the difference map (G(x,y',y))
                l_gxprime = (torch.abs(r_xprime - y_prime)).mean()
                
                #Eq. 5 in the paper, L_{REG}, penalty on the norm of the
                # difference map to make it just produce the necessary changes 
                l_reg = (torch.abs(delta_x)).mean()
                
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
            delta_x = net_g(fixed_x, preprocess_pft_values_for_generator(fixed_yprime), preprocess_pft_values_for_generator(fixed_y))
            output.log_images(epoch_index, net_g, net_r, fixed_x, delta_x)
            
            #iterating through the full validation set
            data_iter_val = iter(healthy_dataloader_val)
            anomaly_data_iter_val = iter(anomaly_dataloader_val)
            m = 0
            n = 0
            while m<len(data_iter_val):
                
                #get validation data
                if n >= len(anomaly_dataloader_val):
                    anomaly_data_iter_val = iter(anomaly_dataloader_val)
                    n = 0
                _ , yprime = anomaly_data_iter_val.next()
                yprime = yprime.cuda()
                n += 1
                x, y = data_iter_val.next()
                x = x.cuda()
                y = y.cuda()
                m+=1
                
                #get validation output
                delta_x = net_g(x, preprocess_pft_values_for_generator(yprime), preprocess_pft_values_for_generator(y))
                
                #calculate normalized cross correlation
                metric.add_ncc(yprime, y, delta_x)
                
            net_r.train()
            net_g.train()
        output.log_added_values(epoch_index, metric)
    
def main():
    #get user options/configurations
    opt = opts.get_opt()
    
    #set cuda visible devices if user specified a value
    if opt.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    
    #load class to save metrics, iamges and models to disk
    output = outputs.Outputs(opt)
    
    #load class to store metrics and losses values
    metric = metrics.Metrics()
    
    #load synthetic dataset
    healthy_dataloader_train, anomaly_dataloader_train = init_synth_dataloader_original(
        opt.folder_toy_dataset, opt.batch_size, mode='train')
    healthy_dataloader_val, anomaly_dataloader_val = init_synth_dataloader_original(
        opt.folder_toy_dataset, opt.batch_size, mode='val')
    
    net_g, net_r = init_model(opt)

    optim_g, optim_r = init_optimizer(opt, net_g=net_g, net_r=net_r)

    net_g = net_g.cuda()
    net_r = net_r.cuda()
    
    train(opt,
          healthy_dataloader_train, anomaly_dataloader_train, healthy_dataloader_val,anomaly_dataloader_val,
          net_g=net_g, net_r=net_r, optim_g=optim_g, optim_r=optim_r, output=output, metric=metric)

if __name__ == '__main__':
    main()
