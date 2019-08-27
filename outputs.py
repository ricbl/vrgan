from tensorboardX import SummaryWriter
import logging
from collections import OrderedDict
import os
from PIL import Image
import numpy as np
import torch
import torchvision

class BatchNormalizeTensorMinMax01(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        maxes = torch.max(torch.max(tensor, dim = 2, keepdim=True)[0], dim = 3, keepdim=True)[0]
        mins = torch.min(torch.min(tensor, dim = 2, keepdim=True)[0], dim = 3, keepdim=True)[0]
        tensor = (tensor-mins)/(maxes - mins)
        tensor = (tensor - 0.5)*2
        return tensor

def save_image(filepath, numpy_array):
    im = Image.fromarray((numpy_array*0.5 + 0.5)*255)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(filepath)

class Outputs():
    def __init__(self, opt):
        output_folder = opt.save_folder+'/'+opt.experiment+'_'+opt.timestamp
        self.output_folder = output_folder
        if not os.path.exists(output_folder ):
            os.mkdir(output_folder)
        logging.basicConfig(filename = output_folder +'/log.txt' ,level = logging.INFO)
        self.log_configs(opt)
        self.writer = SummaryWriter(output_folder + '/tensorboard/')
    
    def log_configs(self, opt):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in sorted(vars(opt).items()):
            logging.info(key + ': ' + str(value).replace('\n', ' ').replace('\r', ''))
        logging.info('-----------------------------end used configs-----------------------------')
            
    def log_fixed(self,fixed_x, fixed_y, fixed_yprime):
        fmi = np.vstack(np.hsplit(np.hstack(BatchNormalizeTensorMinMax01()(fixed_x.detach()).cpu()[:, 0]), 4))
        save_image(self.output_folder+'/real_samples.png', fmi)
        with open(self.output_folder+'/real_samples_gt.txt', 'w') as f:
            out_gt = np.vstack(np.hsplit(np.hstack(fixed_y.cpu()), 4))
            for i in range(out_gt.shape[0]):
                f.write(str( out_gt[i,:]))
                f.write('\n')
        with open(self.output_folder+'/real_samples_desired.txt', 'w') as f:
            out_gt = np.vstack(np.hsplit(np.hstack(fixed_yprime.cpu()), 4))
            for i in range(out_gt.shape[0]):
                f.write(str( out_gt[i,:]))
                f.write('\n')
        
    def log_added_values(self, epoch, metrics):
        averages = metrics.get_average()
        logging.info('Metrics for epoch: ' + str(epoch))
        for key, average in averages.items():
            self.writer.add_scalar(key, average, epoch)
            logging.info(key + ': ' + str(average))
        if ('gen_loss' in averages.keys()) and ('regressor_loss' in averages.keys()):
            self.writer.add_scalar('total_loss', averages['gen_loss']+averages['regressor_loss'],epoch) 
        
    def log_images(self, epoch, net_g, net_r, fixed_x, delta_x):
        torch.save(net_g.state_dict(), '{:}/generator_state_dict_{:05d}'.format(self.output_folder, epoch)) 
        torch.save(net_r.state_dict(), '{:}/regressor_state_dict_{:05d}'.format(self.output_folder, epoch)) 
        inp = np.vstack(np.hsplit(np.hstack(BatchNormalizeTensorMinMax01()(fixed_x.detach() + delta_x.detach()).cpu()[:, 0]), 4))
        img = np.vstack(np.hsplit(np.hstack(delta_x.detach().cpu()[:, 0]), 4))
        path = '{:}/delta_x_samples_{:05d}.png'.format(self.output_folder, epoch)
        save_image(path, img)
        path = '{:}/xprime_samples_{:05d}.png'.format(self.output_folder, epoch)
        save_image(path, inp)
        
    def log_batch(self, epoch, batch_index, metric):
        logging.info('Epoch: ' + str(epoch) + '; Batch ' + str(batch_index) +'; Loss: ' + str(metric.get_last_added_value('gen_loss').item() + metric.get_last_added_value('regressor_loss').item()))
        
    def log_delta_x_gt(self, delta_x_gt):
        img = np.vstack(np.hsplit(np.hstack(delta_x_gt.detach().cpu()[:, 0]), 4))
        path = '{:}/delta_x_gt.png'.format(self.output_folder)
        save_image(path, img)