
import torch
import torchvision
import types

#class to add a preprocessing module to a model
class ClassifierWithPreprocessing(torch.nn.Module):
    def __init__(self, original_model, preprocessing_model):
        super().__init__()
        self.preprocessing_model = preprocessing_model
        self.original_model = original_model
    
    def forward(self, x):
        x = self.preprocessing_model(x)
        x = self.original_model(x)
        return x

# normalizes a batch of tensors according to mean and std
class BatchNormalizeTensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        to_return = (tensor-self.mean)/self.std
        return to_return

# preprocess the inputs of a classifier to normailze them with ImageNet statistics
class ClassifierInputs(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
    
    def forward(self, x):
        return BatchNormalizeTensor(torch.FloatTensor([0.485, 0.456, 0.406]).cuda().view([1,3,1,1]), 
            torch.FloatTensor([0.229, 0.224, 0.225]).cuda().view([1,3,1,1]))((x).expand([-1,3,-1,-1]))

# function that provides with the classifier models used in the paper
def init_model(opt):
    net_r = torchvision.models.resnet18(pretrained = True) 
    n_features_fc = net_r.fc.in_features
    net_r.fc = torch.nn.Linear(in_features = n_features_fc, out_features = 1)
    net_r = ClassifierWithPreprocessing(net_r, ClassifierInputs(opt))
    if opt.load_checkpoint_r is not None:
        net_r.load_state_dict(torch.load(opt.load_checkpoint_r))
    # following Baumgartner et al. (2018), turning off batch normalization 
    # on the critic
    def train(self, mode = True):
        super(type(net_r), self).train(mode)
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d) or \
            isinstance(module, torch.nn.modules.BatchNorm2d) or \
            isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()
    net_r.train = types.MethodType(train, net_r)
    return net_r.cuda()